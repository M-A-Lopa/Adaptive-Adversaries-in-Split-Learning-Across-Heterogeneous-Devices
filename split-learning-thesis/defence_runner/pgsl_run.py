import matplotlib
matplotlib.use('Agg')  

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import pandas as pd
from config import Config
from dataset import DatasetLoader
from all_model.pgsl_models import build_pgsl_client, PGSLServerModel
from all_split_learning.pgsl_split_learning import PGSLSplitLearningTrainer
from all_defences.pgsl_defense import PGSLDefenseModules, AdaptiveWeightedDecisionFusion
from all_attacks.attack_unsplit import UnSplitAttack
from all_attacks.attacks_whitebox import WhiteBoxInversionAttack, AttackMetricsTracker
from all_attacks.ae_decoder_attack import run_ae_decoder_attack, save_ae_attack_visualization
from all_attacks.fsha_attack import FSHAAttack
from all_attacks.label_leakage_attack import GradientNormLabelLeakageAttack, build_binary_split_loaders
from all_attacks.villain_backdoor_attack import VILLAINBackdoorAttack, build_indexed_loader
from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack
from torch.utils.data import DataLoader


def pgsl_preprocess_fn(inputs):
    if Config.DATASET == 'CIFAR10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=inputs.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.2023, 0.1994, 0.2010], device=inputs.device).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.1307], device=inputs.device).view(1, 1, 1, 1)
        std  = torch.tensor([0.3081], device=inputs.device).view(1, 1, 1, 1)
    denorm = torch.clamp(inputs * std + mean, 0.0, 1.0)
    return PGSLDefenseModules.space_to_depth_downsample(denorm, saliency_map=None)


class PGSLClientAdapter(nn.Module):
    def __init__(self, pgsl_client):
        super().__init__()
        self.pgsl_client = pgsl_client

    def forward(self, x):
        return self.pgsl_client(pgsl_preprocess_fn(x))


class PGSLServerAdapter(nn.Module):
    def __init__(self, pgsl_server):
        super().__init__()
        self.pgsl_server = pgsl_server

    def forward(self, smashed_data):
        out_a, out_r, out_f = self.pgsl_server(smashed_data, run_full_pipeline=True)
        return AdaptiveWeightedDecisionFusion.fuse_outputs(out_a, out_r, out_f)


def _smashed_channels_for(base_client, in_channels, image_size, device):
    with torch.no_grad():
        dummy = torch.zeros(1, 4 * in_channels + 1, image_size // 2, image_size // 2, device=device)
        out = base_client.to(device)(dummy)
    return out.shape[1]


def load_or_train_pgsl(device, train_loader, test_loader, in_channels, image_size):
    base_client = build_pgsl_client(original_in_channels=in_channels).to(device)
    smashed_channels = _smashed_channels_for(base_client, in_channels, image_size, device)
    base_server = PGSLServerModel(num_classes=Config.NUM_CLASSES, smashed_channels=smashed_channels).to(device)

    pgsl_ckpt_path    = f"{Config.SAVE_DIR}/best_pgsl_{Config.MODEL_NAME.lower()}_sl_{Config.DATASET}.pth"
    vanilla_ckpt_path = f"{Config.SAVE_DIR}/best_{Config.MODEL_NAME.lower()}_sl_{Config.DATASET}.pth"

    if os.path.exists(pgsl_ckpt_path):
        print(f"\n[✓] Found PGSL checkpoint: {pgsl_ckpt_path}")
        ckpt = torch.load(pgsl_ckpt_path, map_location=device)
        base_client.load_state_dict(ckpt['client_state'])
        base_server.load_state_dict(ckpt['server_state'])
        print(f"    Best PGSL accuracy: {ckpt['best_acc']:.2f}%")

    else:
        if os.path.exists(vanilla_ckpt_path) and Config.MODEL_NAME not in ("KAGN", "PyramidCNN"):
            print(f"\n[!] No PGSL checkpoint found.")
            print(f"[✓] Found vanilla checkpoint: {vanilla_ckpt_path}")
            print("    Adapting vanilla weights to PGSL architecture...")
            vanilla_ckpt = torch.load(vanilla_ckpt_path, map_location=device)
            _adapt_vanilla_weights_to_pgsl(vanilla_ckpt, base_client, base_server, device)
            print("    Starting PGSL training from adapted weights...")
        else:
            print(f"\n[!] No PGSL checkpoint found. Training PGSL from scratch...")

        trainer = PGSLSplitLearningTrainer(client_model=base_client, server_model=base_server,
                                           train_loader=train_loader, test_loader=test_loader)
        trainer.train()
        trainer.save_results()

    return base_client, base_server, smashed_channels


def _adapt_vanilla_weights_to_pgsl(vanilla_ckpt, client_model, server_model, device):
    vanilla_client_state = vanilla_ckpt['client_state']
    vanilla_server_state = vanilla_ckpt['server_state']

    pgsl_client_state = client_model.state_dict()
    old_w = vanilla_client_state['conv1.0.weight']
    new_w = pgsl_client_state['conv1.0.weight']

    if old_w.shape != new_w.shape:
        print(f"  Inflating conv1 weights: {list(old_w.shape)} → {list(new_w.shape)}")
        inflated = torch.zeros(new_w.shape, device=device)
        inflated[:, :old_w.shape[1], :, :] = old_w
        vanilla_client_state['conv1.0.weight'] = inflated

    client_model.load_state_dict(vanilla_client_state)
    print("  Client weights adapted and loaded.")

    pgsl_server_state = server_model.state_dict()

    for stream in ['a', 'r', 'f']:
        mapping = {
            'conv3.0.weight': f'stream_{stream}.0.weight',
            'conv3.0.bias': f'stream_{stream}.0.bias',
            'conv3.1.weight': f'stream_{stream}.1.weight',
            'conv3.1.bias': f'stream_{stream}.1.bias',
            'conv3.1.running_mean': f'stream_{stream}.1.running_mean',
            'conv3.1.running_var': f'stream_{stream}.1.running_var',
            'conv3.1.num_batches_tracked': f'stream_{stream}.1.num_batches_tracked',
            'fc.1.weight': f'classifier_{stream}.1.weight',
            'fc.1.bias': f'classifier_{stream}.1.bias',
            'fc.4.weight': f'classifier_{stream}.4.weight',
            'fc.4.bias': f'classifier_{stream}.4.bias',
        }
        for vanilla_key, pgsl_key in mapping.items():
            if vanilla_key in vanilla_server_state and pgsl_key in pgsl_server_state:
                if vanilla_server_state[vanilla_key].shape == pgsl_server_state[pgsl_key].shape:
                    pgsl_server_state[pgsl_key] = vanilla_server_state[vanilla_key]

    server_model.load_state_dict(pgsl_server_state)
    print("  Server weights mapped (stream_a initialized from vanilla conv3+fc).")
    print("  Streams r and f are randomly initialized.")


if __name__ == "__main__":

    print("="*60)
    print("  PGSL DEFENSE EXPERIMENT")
    print(f"  Model   : {Config.MODEL_NAME}")
    print(f"  Dataset : {Config.DATASET}")
    print(f"  Cut layer: {Config.CUT_LAYER}")
    print("="*60)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"  Device  : {device}\n")

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()

    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    image_size  = 28 if Config.DATASET == 'MNIST' else 32

    base_client, base_server, smashed_channels = load_or_train_pgsl(
        device, train_loader, test_loader, in_channels, image_size
    )

    client_model = PGSLClientAdapter(base_client).to(device)
    server_model = PGSLServerAdapter(base_server).to(device)

    def build_fresh_client():
        base = build_pgsl_client(original_in_channels=in_channels).to(device)
        return PGSLClientAdapter(base).to(device)

    def build_fresh_split(num_classes):
        b_client = build_pgsl_client(original_in_channels=in_channels).to(device)
        b_server = PGSLServerModel(num_classes=num_classes, smashed_channels=smashed_channels).to(device)
        return PGSLClientAdapter(b_client).to(device), PGSLServerAdapter(b_server).to(device)

    if not Config.RUN_ATTACK:
        print("\n  Config.RUN_ATTACK is False -- skipping attack suite.")
        sys.exit(0)

    MAX_IMAGES  = 32
    ITERATIONS  = 1000
    HIJACK_EPOCHS    = 50
    CRITIC_ITERS     = 5
    LEAKAGE_EPOCHS   = 5
    POSITIVE_CLASS   = 0
    POSITIVE_RATIO   = 0.1
    LEAKAGE_BATCH    = 128
    LEAKAGE_LR       = 1e-4
    WARMUP_EPOCHS    = 15
    INFERENCE_EPOCHS = 5
    INJECTION_EPOCHS = 15
    VILLAIN_BATCH    = 32
    TARGET_LABEL     = 0
    TRIGGER_BETA     = 1.0
    TRIGGER_FRACTION = 0.5
    DROPOUT_KEEP     = 0.75
    GAMMA_LOW        = 0.6
    GAMMA_HIGH       = 1.2
    POISON_RATE      = 0.05
    CANDIDATES       = 14
    BACKDOOR_TARGET_LABEL     = 0
    BACKDOOR_POISON_RATE      = 0.05
    BACKDOOR_PATCH_SIZE       = 4
    BACKDOOR_TRIGGER_VALUE    = 1.0
    BACKDOOR_TRAIN_EPOCHS     = 10
    BACKDOOR_SURROGATE_EPOCHS = 5

    if Config.DATASET == 'CIFAR10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(device)
        std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(device)
    else:
        mean = torch.tensor([0.1307]).view(1,1,1,1).to(device)
        std  = torch.tensor([0.3081]).view(1,1,1,1).to(device)

    all_results = {}

    # Attack 1: White-Box
    print("\n" + "="*60)
    print(f"  WHITE-BOX ATTACK vs PGSL — {Config.MODEL_NAME}")
    print("="*60)

    attacker = WhiteBoxInversionAttack(client_model=client_model, dataset=Config.DATASET, iterations=ITERATIONS, lr=1e-2)
    tracker = AttackMetricsTracker()
    images_processed = 0

    for inputs, _ in test_loader:
        if images_processed >= MAX_IMAGES:
            break
        inputs    = inputs.to(device)
        remaining = MAX_IMAGES - images_processed
        inputs    = inputs[:remaining]

        with torch.no_grad():
            target_smashed = client_model(inputs)
        reconstructed = attacker.reconstruct(target_smashed, inputs.shape)

        inputs_denorm = torch.clamp(inputs * std + mean, 0, 1)
        tracker.log_batch(inputs_denorm, reconstructed)
        images_processed += inputs.shape[0]

    wb_summary = tracker.get_summary()
    all_results['WhiteBox'] = wb_summary
    print(f"\n  PSNR: {wb_summary['mean_psnr']:.2f} dB | SSIM: {wb_summary['mean_ssim']:.4f}")
    pd.DataFrame([wb_summary]).to_csv(
        f"{Config.RESULTS_DIR}/pgsl_attack_whitebox_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    # Attack 2: UnSplit
    print("\n" + "="*60)
    print(f"  UNSPLIT ATTACK vs PGSL — {Config.MODEL_NAME}")
    print("="*60)

    unsplit_attacker = UnSplitAttack(
        client_model=client_model, in_channels=in_channels, clone_builder=build_fresh_client,
        main_iters=Config.unsplit_main_iters, input_iters=Config.unsplit_input_iters, model_iters=Config.unsplit_model_iters
    )
    unsplit_summary = unsplit_attacker.run_attack(test_loader, num_batches=MAX_IMAGES // 32 or 1)
    all_results['UnSplit'] = unsplit_summary
    print(f"\n  PSNR: {unsplit_summary['psnr']:.2f} dB | SSIM: {unsplit_summary['ssim']:.4f}")
    pd.DataFrame([unsplit_summary]).to_csv(
        f"{Config.RESULTS_DIR}/pgsl_attack_unsplit_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    # Attack 3: AE Decoder
    print("\n" + "="*60)
    print(f"  AE DECODER ATTACK vs PGSL — {Config.MODEL_NAME}")
    print("="*60)

    ae_summary, ae_orig, ae_recon = run_ae_decoder_attack(
        client_model=client_model, train_loader=train_loader, test_loader=test_loader, device=device,
        dataset=Config.DATASET, preprocess_fn=None, ae_epochs=50, label=f'{Config.MODEL_NAME} PGSL AE Decoder'
    )
    all_results['AE_Decoder'] = ae_summary
    pd.DataFrame([ae_summary]).to_csv(
        f"{Config.RESULTS_DIR}/pgsl_attack_ae_decoder_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    if len(ae_orig) > 0:
        save_ae_attack_visualization(
            originals=ae_orig, reconstructed=ae_recon, baseline_summary=wb_summary, defense_summary=ae_summary,
            defense_name=f"PGSL-{Config.MODEL_NAME}", dataset=Config.DATASET, results_dir=Config.RESULTS_DIR
        )

    # Attack 4: FSHA
    print("\n" + "="*60)
    print(f"  FSHA ATTACK vs PGSL — {Config.MODEL_NAME}")
    print("="*60)

    fsha_client = build_fresh_client().to(device)
    fsha_private_loader = DataLoader(train_loader.dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    fsha_public_loader  = DataLoader(test_loader.dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    fsha_attacker = FSHAAttack(
        client_model=fsha_client, in_channels=in_channels, dataset=Config.DATASET,
        pilot_builder=build_fresh_client, critic_iters=CRITIC_ITERS
    )
    fsha_attacker.hijack(fsha_private_loader, fsha_public_loader, epochs=HIJACK_EPOCHS)
    fsha_summary = fsha_attacker.reconstruct(train_loader, num_images=MAX_IMAGES)
    all_results['FSHA'] = fsha_summary

    fsha_source = f"{Config.RESULTS_DIR}/fsha_no_defense.png"
    fsha_target = f"{Config.RESULTS_DIR}/fsha_pgsl_{Config.MODEL_NAME.lower()}_{Config.DATASET}.png"
    if os.path.exists(fsha_source):
        if os.path.exists(fsha_target):
            os.remove(fsha_target)
        os.rename(fsha_source, fsha_target)

    pd.DataFrame([fsha_summary]).to_csv(
        f"{Config.RESULTS_DIR}/pgsl_attack_fsha_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    # Attack 5: Label Leakage
    print("\n" + "="*60)
    print(f"  LABEL LEAKAGE ATTACK vs PGSL — {Config.MODEL_NAME}")
    print("="*60)

    binary_train_loader, binary_test_loader = build_binary_split_loaders(
        train_loader.dataset, target_class=POSITIVE_CLASS, positive_ratio=POSITIVE_RATIO, batch_size=LEAKAGE_BATCH
    )

    leakage_client, leakage_server = build_fresh_split(num_classes=1)

    leakage_attacker = GradientNormLabelLeakageAttack(
        client_model=leakage_client, server_model=leakage_server, dataset=Config.DATASET,
        target_class=POSITIVE_CLASS, learning_rate=LEAKAGE_LR
    )
    leakage_summary = leakage_attacker.run(binary_train_loader, binary_test_loader, epochs=LEAKAGE_EPOCHS)
    leakage_attacker.save_visualization(tag=f"pgsl_{Config.MODEL_NAME.lower()}_{Config.DATASET}")
    all_results['LabelLeakage'] = leakage_summary

    pd.DataFrame([leakage_summary]).to_csv(
        f"{Config.RESULTS_DIR}/pgsl_attack_label_leakage_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)
    pd.DataFrame(leakage_attacker.history).to_csv(
        f"{Config.RESULTS_DIR}/pgsl_label_leakage_batches_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)
    
    
    # Attack 6: VILLAIN
    print("\n" + "="*60)
    print(f"  VILLAIN BACKDOOR ATTACK vs PGSL — {Config.MODEL_NAME}")
    print("="*60)

    indexed_loader = build_indexed_loader(train_loader.dataset, batch_size=VILLAIN_BATCH, shuffle=True)
    villain_client, villain_server = build_fresh_split(num_classes=Config.NUM_CLASSES)

    villain_attacker = VILLAINBackdoorAttack(
        client_model=villain_client, server_model=villain_server, base_dataset=train_loader.dataset,
        dataset=Config.DATASET, num_classes=Config.NUM_CLASSES, target_label=TARGET_LABEL, beta=TRIGGER_BETA,
        trigger_fraction=TRIGGER_FRACTION, dropout_keep=DROPOUT_KEEP, gamma_low=GAMMA_LOW, gamma_high=GAMMA_HIGH,
        poison_rate=POISON_RATE, candidates_per_batch=CANDIDATES
    )

    villain_attacker.warmup(indexed_loader, epochs=WARMUP_EPOCHS)
    villain_baseline, _ = villain_attacker.evaluate(test_loader)
    print(f"  Clean data accuracy before attack: {villain_baseline:.2f}%")

    villain_attacker.infer_labels(indexed_loader, epochs=INFERENCE_EPOCHS)
    villain_attacker.fabricate_trigger(indexed_loader)
    villain_attacker.inject_backdoor(indexed_loader, test_loader, epochs=INJECTION_EPOCHS)

    villain_summary = villain_attacker.summarise(test_loader=test_loader, clean_baseline=villain_baseline)
    villain_attacker.save_visualization(tag=f"pgsl_{Config.MODEL_NAME.lower()}_{Config.DATASET}")
    all_results['VILLAIN'] = villain_summary

    pd.DataFrame([villain_summary]).to_csv(
        f"{Config.RESULTS_DIR}/pgsl_attack_villain_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)
    pd.DataFrame(villain_attacker.history).to_csv(
        f"{Config.RESULTS_DIR}/pgsl_villain_epochs_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    # Attack 7: Backdoor Poisoning -- Client-Side
    print("\n" + "="*60)
    print(f"  BACKDOOR POISONING (CLIENT-SIDE) vs PGSL — {Config.MODEL_NAME}")
    print("="*60)

    backdoor_c_client, backdoor_c_server = build_fresh_split(num_classes=Config.NUM_CLASSES)
    backdoor_client_attacker = BackdoorPoisonAttack(
        client_model=backdoor_c_client, server_model=backdoor_c_server, base_dataset=train_loader.dataset,
        dataset=Config.DATASET, num_classes=Config.NUM_CLASSES, mode='client', target_label=BACKDOOR_TARGET_LABEL,
        poison_rate=BACKDOOR_POISON_RATE, patch_size=BACKDOOR_PATCH_SIZE, trigger_value=BACKDOOR_TRIGGER_VALUE,
        model_tag=f"pgsl_{Config.MODEL_NAME.lower()}_sl",
    )
    backdoor_client_attacker.train(train_loader, test_loader, epochs=BACKDOOR_TRAIN_EPOCHS)
    backdoor_client_summary = backdoor_client_attacker.summarise()
    backdoor_client_attacker.save_visualization(tag=f"pgsl_{Config.MODEL_NAME.lower()}_{Config.DATASET}")
    all_results['BackdoorPoison_Client'] = backdoor_client_summary

    pd.DataFrame([backdoor_client_summary]).to_csv(
        f"{Config.RESULTS_DIR}/pgsl_attack_backdoor_poison_client_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)
    pd.DataFrame(backdoor_client_attacker.history).to_csv(
        f"{Config.RESULTS_DIR}/pgsl_backdoor_poison_client_epochs_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    # Attack 8: Backdoor Poisoning -- Server-Side
    print("\n" + "="*60)
    print(f"  BACKDOOR POISONING (SERVER-SIDE) vs PGSL — {Config.MODEL_NAME}")
    print("="*60)

    backdoor_s_client, backdoor_s_server = build_fresh_split(num_classes=Config.NUM_CLASSES)
    backdoor_server_attacker = BackdoorPoisonAttack(
        client_model=backdoor_s_client, server_model=backdoor_s_server, base_dataset=train_loader.dataset,
        dataset=Config.DATASET, num_classes=Config.NUM_CLASSES, mode='server', target_label=BACKDOOR_TARGET_LABEL,
        poison_rate=BACKDOOR_POISON_RATE, patch_size=BACKDOOR_PATCH_SIZE, trigger_value=BACKDOOR_TRIGGER_VALUE,
        surrogate_builder=build_fresh_client, model_tag=f"pgsl_{Config.MODEL_NAME.lower()}_sl",
    )
    backdoor_server_attacker.pretrain_server_backdoor(test_loader, epochs=BACKDOOR_SURROGATE_EPOCHS)
    backdoor_server_attacker.train(train_loader, test_loader, epochs=BACKDOOR_TRAIN_EPOCHS)
    backdoor_server_summary = backdoor_server_attacker.summarise()
    backdoor_server_attacker.save_visualization(tag=f"pgsl_{Config.MODEL_NAME.lower()}_{Config.DATASET}")
    all_results['BackdoorPoison_Server'] = backdoor_server_summary

    pd.DataFrame([backdoor_server_summary]).to_csv(
        f"{Config.RESULTS_DIR}/pgsl_attack_backdoor_poison_server_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)
    pd.DataFrame(backdoor_server_attacker.history).to_csv(
        f"{Config.RESULTS_DIR}/pgsl_backdoor_poison_server_epochs_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    # Combined summary
    blank = "—"
    print("\n" + "="*94)
    print(f"   ALL ATTACKS vs PGSL SUMMARY — {Config.MODEL_NAME} on {Config.DATASET}")
    print("="*94)
    print(f"{'Attack':<20} {'PSNR (dB)':>11} {'SSIM':>9} {'MSE':>9} {'Leak AUC':>10} {'LIA (%)':>9} {'ASR (%)':>9} {'CDA (%)':>9}")
    print("-"*94)
    print(f"  {'White-Box':<18} {wb_summary['mean_psnr']:>11.2f} {wb_summary['mean_ssim']:>9.4f} {blank:>9} {blank:>10} {blank:>9} {blank:>9} {blank:>9}")
    print(f"  {'UnSplit':<18} {unsplit_summary['psnr']:>11.2f} {unsplit_summary['ssim']:>9.4f} {blank:>9} {blank:>10} {blank:>9} {blank:>9} {blank:>9}")
    print(f"  {'AE Decoder':<18} {ae_summary['mean_psnr']:>11.2f} {ae_summary['mean_ssim']:>9.4f} {blank:>9} {blank:>10} {blank:>9} {blank:>9} {blank:>9}")
    print(f"  {'FSHA':<18} {fsha_summary['psnr']:>11.2f} {fsha_summary['ssim']:>9.4f} {fsha_summary['mse']:>9.5f} {blank:>10} {blank:>9} {blank:>9} {blank:>9}")
    print(f"  {'Label Leakage':<18} {blank:>11} {blank:>9} {blank:>9} {leakage_summary['q95_norm_leak_auc_cut']:>10.4f} {blank:>9} {blank:>9} {leakage_summary['test_auc']:>9.4f}")
    print(f"  {'VILLAIN':<18} {blank:>11} {blank:>9} {blank:>9} {blank:>10} {villain_summary['lia']:>9.2f} {villain_summary['asr']:>9.2f} {villain_summary['cda']:>9.2f}")
    print(f"  {'Backdoor(Client)':<18} {blank:>11} {blank:>9} {blank:>9} {blank:>10} {blank:>9} {backdoor_client_summary['asr']:>9.2f} {backdoor_client_summary['cda']:>9.2f}")
    print(f"  {'Backdoor(Server)':<18} {blank:>11} {blank:>9} {blank:>9} {blank:>10} {blank:>9} {backdoor_server_summary['asr']:>9.2f} {backdoor_server_summary['cda']:>9.2f}")
    print("="*94)

    combined_path = f"{Config.RESULTS_DIR}/pgsl_all_attacks_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv"
    pd.DataFrame({
        'attack': ['WhiteBox', 'UnSplit', 'AE_Decoder', 'FSHA', 'LabelLeakage', 'VILLAIN', 'BackdoorPoison_Client', 'BackdoorPoison_Server'],
        'psnr': [wb_summary['mean_psnr'], unsplit_summary['psnr'], ae_summary['mean_psnr'], fsha_summary['psnr'], None, None, None, None],
        'ssim': [wb_summary['mean_ssim'], unsplit_summary['ssim'], ae_summary['mean_ssim'], fsha_summary['ssim'], None, None, None, None],
        'mse': [None, None, None, fsha_summary['mse'], None, None, None, None],
        'norm_leak_auc_cut': [None, None, None, None, leakage_summary['q95_norm_leak_auc_cut'], None, None, None],
        'lia': [None, None, None, None, leakage_summary['q95_majority_accuracy_cut'], villain_summary['lia'], None, None],
        'asr': [None, None, None, None, None, villain_summary['asr'], backdoor_client_summary['asr'], backdoor_server_summary['asr']],
        'cda': [None, None, None, None, None, villain_summary['cda'], backdoor_client_summary['cda'], backdoor_server_summary['cda']],
        'label_test_auc': [None, None, None, None, leakage_summary['test_auc'], None, None, None],
    }).to_csv(combined_path, index=False)
    print(f"\n  Combined PGSL results saved → {combined_path}")