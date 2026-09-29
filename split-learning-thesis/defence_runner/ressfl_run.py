import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from dataset import DatasetLoader
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel
from all_split_learning.ressfl_split_learning import ResSFLTrainer
from all_attacks.attack_unsplit import UnSplitAttack
from all_attacks.attacks_whitebox import WhiteBoxInversionAttack, AttackMetricsTracker
from all_attacks.ae_decoder_attack import run_ae_decoder_attack, save_ae_attack_visualization
from all_attacks.fsha_attack import FSHAAttack
from all_attacks.label_leakage_attack import GradientNormLabelLeakageAttack, build_binary_split_loaders
from all_attacks.villain_backdoor_attack import VILLAINBackdoorAttack, build_indexed_loader
from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack
from torch.utils.data import DataLoader


def build_client_server(num_classes, device, in_channels):
    if Config.MODEL_NAME == "KAGN":
        client = KAGNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels, degree=Config.DEGREE).to(device)
        server = KAGNServerModel(cut_layer=Config.CUT_LAYER, num_classes=num_classes, in_channels=in_channels, degree=Config.DEGREE).to(device)
    elif Config.MODEL_NAME == "PyramidCNN":
        client = PyramidCNNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels).to(device)
        server = PyramidCNNServerModel(cut_layer=Config.CUT_LAYER, num_classes=num_classes, in_channels=in_channels).to(device)
    else:
        client = ClientModel(in_channels=in_channels).to(device)
        server = ServerModel(num_classes=num_classes).to(device)
    return client, server


def load_or_train_ressfl(device, train_loader, test_loader):
    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    client_model, server_model = build_client_server(Config.NUM_CLASSES, device, in_channels)

    ressfl_ckpt  = f"{Config.SAVE_DIR}/best_ressfl_{Config.MODEL_NAME.lower()}_{Config.DATASET}.pth"
    vanilla_ckpt = f"{Config.SAVE_DIR}/best_{Config.MODEL_NAME.lower()}_sl_{Config.DATASET}.pth"

    if os.path.exists(ressfl_ckpt):
        print(f"\n[✓] Found ResSFL checkpoint: {ressfl_ckpt}")
        ckpt = torch.load(ressfl_ckpt, map_location=device)
        client_model.load_state_dict(ckpt['client_state'])
        server_model.load_state_dict(ckpt['server_state'])
        print(f"    Best ResSFL accuracy: {ckpt['best_acc']:.2f}%")
        print(f"    SSIM threshold used : {ckpt.get('ssim_threshold', 0.4)}")

    elif os.path.exists(vanilla_ckpt):
        print(f"\n[!] No ResSFL checkpoint found.")
        print(f"[✓] Found vanilla checkpoint: {vanilla_ckpt}")
        print("    Initializing client+server from vanilla weights (same architecture)...")
        vanilla = torch.load(vanilla_ckpt, map_location=device)
        client_model.load_state_dict(vanilla['client_state'])
        server_model.load_state_dict(vanilla['server_state'])
        print(f"    Vanilla accuracy: {vanilla.get('best_acc', float('nan')):.2f}%")
        print("    Starting ResSFL adversarial training from vanilla init...\n")

        trainer = ResSFLTrainer(client_model=client_model, server_model=server_model, train_loader=train_loader, test_loader=test_loader)
        trainer.train()
        trainer.save_results()

    else:
        print(f"\n[!] No checkpoint found. Training ResSFL from scratch...")
        trainer = ResSFLTrainer(client_model=client_model, server_model=server_model, train_loader=train_loader, test_loader=test_loader)
        trainer.train()
        trainer.save_results()

    return client_model, server_model


if __name__ == "__main__":

    print("="*60)
    print("  RESSFL DEFENSE EXPERIMENT")
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

    client_model, server_model = load_or_train_ressfl(device, train_loader, test_loader)

    def build_fresh_client():
        c, _ = build_client_server(Config.NUM_CLASSES, device, in_channels)
        return c

    def build_fresh_split(num_classes):
        return build_client_server(num_classes, device, in_channels)

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
    print(f"  WHITE-BOX ATTACK vs RESSFL — {Config.MODEL_NAME}")
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
        f"{Config.RESULTS_DIR}/ressfl_attack_whitebox_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    # Attack 2: UnSplit
    print("\n" + "="*60)
    print(f"  UNSPLIT ATTACK vs RESSFL — {Config.MODEL_NAME}")
    print("="*60)

    unsplit_attacker = UnSplitAttack(
        client_model=client_model, in_channels=in_channels, clone_builder=build_fresh_client,
        main_iters=Config.unsplit_main_iters, input_iters=Config.unsplit_input_iters, model_iters=Config.unsplit_model_iters
    )
    unsplit_summary = unsplit_attacker.run_attack(test_loader, num_batches=MAX_IMAGES // 32 or 1)
    all_results['UnSplit'] = unsplit_summary
    print(f"\n  PSNR: {unsplit_summary['psnr']:.2f} dB | SSIM: {unsplit_summary['ssim']:.4f}")
    pd.DataFrame([unsplit_summary]).to_csv(
        f"{Config.RESULTS_DIR}/ressfl_attack_unsplit_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    # Attack 3: AE Decoder
    print("\n" + "="*60)
    print(f"  AE DECODER ATTACK vs RESSFL — {Config.MODEL_NAME}")
    print("="*60)

    ae_summary, ae_orig, ae_recon = run_ae_decoder_attack(
        client_model=client_model, train_loader=train_loader, test_loader=test_loader, device=device,
        dataset=Config.DATASET, preprocess_fn=None, ae_epochs=50, label=f'{Config.MODEL_NAME} ResSFL AE Decoder'
    )
    all_results['AE_Decoder'] = ae_summary
    pd.DataFrame([ae_summary]).to_csv(
        f"{Config.RESULTS_DIR}/ressfl_attack_ae_decoder_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    if len(ae_orig) > 0:
        save_ae_attack_visualization(
            originals=ae_orig, reconstructed=ae_recon, baseline_summary=wb_summary, defense_summary=ae_summary,
            defense_name=f"ResSFL-{Config.MODEL_NAME}", dataset=Config.DATASET, results_dir=Config.RESULTS_DIR
        )

    # Attack 4: FSHA
    print("\n" + "="*60)
    print(f"  FSHA ATTACK vs RESSFL — {Config.MODEL_NAME}")
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
    fsha_target = f"{Config.RESULTS_DIR}/fsha_ressfl_{Config.MODEL_NAME.lower()}_{Config.DATASET}.png"
    if os.path.exists(fsha_source):
        if os.path.exists(fsha_target):
            os.remove(fsha_target)
        os.rename(fsha_source, fsha_target)

    pd.DataFrame([fsha_summary]).to_csv(
        f"{Config.RESULTS_DIR}/ressfl_attack_fsha_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    # Attack 5: Label Leakage
    print("\n" + "="*60)
    print(f"  LABEL LEAKAGE ATTACK vs RESSFL — {Config.MODEL_NAME}")
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
    leakage_attacker.save_visualization(tag=f"ressfl_{Config.MODEL_NAME.lower()}_{Config.DATASET}")
    all_results['LabelLeakage'] = leakage_summary

    pd.DataFrame([leakage_summary]).to_csv(
        f"{Config.RESULTS_DIR}/ressfl_attack_label_leakage_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)
    pd.DataFrame(leakage_attacker.history).to_csv(
        f"{Config.RESULTS_DIR}/ressfl_label_leakage_batches_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    # Attack 6: VILLAIN
    print("\n" + "="*60)
    print(f"  VILLAIN BACKDOOR ATTACK vs RESSFL — {Config.MODEL_NAME}")
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
    villain_attacker.save_visualization(tag=f"ressfl_{Config.MODEL_NAME.lower()}_{Config.DATASET}")
    all_results['VILLAIN'] = villain_summary

    pd.DataFrame([villain_summary]).to_csv(
        f"{Config.RESULTS_DIR}/ressfl_attack_villain_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)
    pd.DataFrame(villain_attacker.history).to_csv(
        f"{Config.RESULTS_DIR}/ressfl_villain_epochs_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    # Attack 7: Backdoor Poisoning -- Client-Side
    print("\n" + "="*60)
    print(f"  BACKDOOR POISONING (CLIENT-SIDE) vs RESSFL — {Config.MODEL_NAME}")
    print("="*60)

    backdoor_c_client, backdoor_c_server = build_fresh_split(num_classes=Config.NUM_CLASSES)
    backdoor_client_attacker = BackdoorPoisonAttack(
        client_model=backdoor_c_client, server_model=backdoor_c_server, base_dataset=train_loader.dataset,
        dataset=Config.DATASET, num_classes=Config.NUM_CLASSES, mode='client', target_label=BACKDOOR_TARGET_LABEL,
        poison_rate=BACKDOOR_POISON_RATE, patch_size=BACKDOOR_PATCH_SIZE, trigger_value=BACKDOOR_TRIGGER_VALUE,
        model_tag=f"ressfl_{Config.MODEL_NAME.lower()}_sl",
    )
    backdoor_client_attacker.train(train_loader, test_loader, epochs=BACKDOOR_TRAIN_EPOCHS)
    backdoor_client_summary = backdoor_client_attacker.summarise()
    backdoor_client_attacker.save_visualization(tag=f"ressfl_{Config.MODEL_NAME.lower()}_{Config.DATASET}")
    all_results['BackdoorPoison_Client'] = backdoor_client_summary

    pd.DataFrame([backdoor_client_summary]).to_csv(
        f"{Config.RESULTS_DIR}/ressfl_attack_backdoor_poison_client_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)
    pd.DataFrame(backdoor_client_attacker.history).to_csv(
        f"{Config.RESULTS_DIR}/ressfl_backdoor_poison_client_epochs_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    # Attack 8: Backdoor Poisoning -- Server-Side
    print("\n" + "="*60)
    print(f"  BACKDOOR POISONING (SERVER-SIDE) vs RESSFL — {Config.MODEL_NAME}")
    print("="*60)

    backdoor_s_client, backdoor_s_server = build_fresh_split(num_classes=Config.NUM_CLASSES)
    backdoor_server_attacker = BackdoorPoisonAttack(
        client_model=backdoor_s_client, server_model=backdoor_s_server, base_dataset=train_loader.dataset,
        dataset=Config.DATASET, num_classes=Config.NUM_CLASSES, mode='server', target_label=BACKDOOR_TARGET_LABEL,
        poison_rate=BACKDOOR_POISON_RATE, patch_size=BACKDOOR_PATCH_SIZE, trigger_value=BACKDOOR_TRIGGER_VALUE,
        surrogate_builder=build_fresh_client, model_tag=f"ressfl_{Config.MODEL_NAME.lower()}_sl",
    )
    backdoor_server_attacker.pretrain_server_backdoor(test_loader, epochs=BACKDOOR_SURROGATE_EPOCHS)
    backdoor_server_attacker.train(train_loader, test_loader, epochs=BACKDOOR_TRAIN_EPOCHS)
    backdoor_server_summary = backdoor_server_attacker.summarise()
    backdoor_server_attacker.save_visualization(tag=f"ressfl_{Config.MODEL_NAME.lower()}_{Config.DATASET}")
    all_results['BackdoorPoison_Server'] = backdoor_server_summary

    pd.DataFrame([backdoor_server_summary]).to_csv(
        f"{Config.RESULTS_DIR}/ressfl_attack_backdoor_poison_server_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)
    pd.DataFrame(backdoor_server_attacker.history).to_csv(
        f"{Config.RESULTS_DIR}/ressfl_backdoor_poison_server_epochs_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    # Combined summary
    blank = "—"
    print("\n" + "="*94)
    print(f"   ALL ATTACKS vs RESSFL SUMMARY — {Config.MODEL_NAME} on {Config.DATASET}")
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

    combined_path = f"{Config.RESULTS_DIR}/ressfl_all_attacks_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv"
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
    print(f"\n  Combined ResSFL results saved → {combined_path}")