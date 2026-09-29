import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from dataset import DatasetLoader
from all_split_learning.unified_split_learning import DefendedSplitLearningTrainer
from all_attacks.attack_unsplit import UnSplitAttack
from all_attacks.attacks_whitebox import WhiteBoxInversionAttack, AttackMetricsTracker
from all_attacks.ae_decoder_attack import run_ae_decoder_attack, save_ae_attack_visualization
from all_attacks.fsha_attack import FSHAAttack
from all_attacks.label_leakage_attack import GradientNormLabelLeakageAttack, build_binary_split_loaders
from all_attacks.villain_backdoor_attack import VILLAINBackdoorAttack, build_indexed_loader
from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel
from all_defences.unified_defense import DefendedClientModel, DefendedServerModel


ALL_ATTACKS = ['whitebox', 'unsplit', 'ae_decoder', 'fsha', 'label_leakage', 'villain', 'backdoor_client', 'backdoor_server']

ABLATION_CONFIGS = {
    'baseline': (
        None,  
        None,
        ALL_ATTACKS,
    ),
    'full_defense': (
        dict(use_augmentation=True, use_vel_norm=True, use_persample_norm=True, use_clip=True, use_gpi=True),
        dict(use_grad_defense=True, use_server_sanitization=True, decorr_weight=0.05, dcor_weight=0.02),
        ALL_ATTACKS,
    ),
    'no_vel_norm': (
        dict(use_augmentation=True, use_vel_norm=False, use_persample_norm=True, use_clip=True, use_gpi=True),
        dict(use_grad_defense=True, use_server_sanitization=True, decorr_weight=0.05, dcor_weight=0.02),
        ['villain'],
    ),
    'no_persample_norm': (
        dict(use_augmentation=True, use_vel_norm=True, use_persample_norm=False, use_clip=True, use_gpi=True),
        dict(use_grad_defense=True, use_server_sanitization=True, decorr_weight=0.05, dcor_weight=0.02),
        ['villain'],
    ),
    'no_clip': (
        dict(use_augmentation=True, use_vel_norm=True, use_persample_norm=True, use_clip=False, use_gpi=True),
        dict(use_grad_defense=True, use_server_sanitization=True, decorr_weight=0.05, dcor_weight=0.02),
        ['villain'],
    ),
    # Label Leakage-targeted
    'no_gpi': (
        dict(use_augmentation=True, use_vel_norm=True, use_persample_norm=True, use_clip=True, use_gpi=False),
        dict(use_grad_defense=True, use_server_sanitization=True, decorr_weight=0.05, dcor_weight=0.02),
        ['label_leakage'],
    ),
    'no_grad_defense': (
        dict(use_augmentation=True, use_vel_norm=True, use_persample_norm=True, use_clip=True, use_gpi=True),
        dict(use_grad_defense=False, use_server_sanitization=True, decorr_weight=0.05, dcor_weight=0.02),
        ['label_leakage'],
    ),
    # Backdoor(Client)-targeted
    'no_augmentation': (
        dict(use_augmentation=False, use_vel_norm=True, use_persample_norm=True, use_clip=True, use_gpi=True),
        dict(use_grad_defense=True, use_server_sanitization=True, decorr_weight=0.05, dcor_weight=0.02),
        ['backdoor_client'],
    ),
    'no_server_sanitization': (
        dict(use_augmentation=True, use_vel_norm=True, use_persample_norm=True, use_clip=True, use_gpi=True),
        dict(use_grad_defense=True, use_server_sanitization=False, decorr_weight=0.05, dcor_weight=0.02),
        ['backdoor_client'],
    ),
    # AE Decoder / FSHA-targeted
    'no_decorr': (
        dict(use_augmentation=True, use_vel_norm=True, use_persample_norm=True, use_clip=True, use_gpi=True),
        dict(use_grad_defense=True, use_server_sanitization=True, decorr_weight=0.0, dcor_weight=0.02),
        ['ae_decoder', 'fsha'],
    ),
    'no_dcor': (
        dict(use_augmentation=True, use_vel_norm=True, use_persample_norm=True, use_clip=True, use_gpi=True),
        dict(use_grad_defense=True, use_server_sanitization=True, decorr_weight=0.05, dcor_weight=0.0),
        ['fsha'],
    ),
}


def build_base_client_server(device, in_channels):
    if Config.MODEL_NAME == "KAGN":
        client = KAGNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels, degree=Config.DEGREE).to(device)
        server = KAGNServerModel(cut_layer=Config.CUT_LAYER, num_classes=Config.NUM_CLASSES, in_channels=in_channels, degree=Config.DEGREE).to(device)
    elif Config.MODEL_NAME == "PyramidCNN":
        client = PyramidCNNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels).to(device)
        server = PyramidCNNServerModel(cut_layer=Config.CUT_LAYER, num_classes=Config.NUM_CLASSES, in_channels=in_channels).to(device)
    else:
        client = ClientModel(in_channels=in_channels).to(device)
        server = ServerModel(num_classes=Config.NUM_CLASSES).to(device)
    return client, server


def build_config_models(config_name, device, in_channels, image_size, num_classes=None):
    """Returns (client_model, server_model) for one ablation config,
    wrapped in the defense (or not, for baseline)."""
    num_classes = num_classes or Config.NUM_CLASSES
    client_kwargs, server_kwargs, _ = ABLATION_CONFIGS[config_name]

    base_client, base_server = build_base_client_server(device, in_channels)
    if num_classes != Config.NUM_CLASSES:
        if Config.MODEL_NAME == "KAGN":
            base_server = KAGNServerModel(cut_layer=Config.CUT_LAYER, num_classes=num_classes, in_channels=in_channels, degree=Config.DEGREE).to(device)
        elif Config.MODEL_NAME == "PyramidCNN":
            base_server = PyramidCNNServerModel(cut_layer=Config.CUT_LAYER, num_classes=num_classes, in_channels=in_channels).to(device)
        else:
            base_server = ServerModel(num_classes=num_classes).to(device)

    if client_kwargs is None:
        return base_client, base_server

    client_model = DefendedClientModel(
        base_client, in_channels, image_size, device,
        use_augmentation=client_kwargs['use_augmentation'],
        use_vel_norm=client_kwargs['use_vel_norm'],
        use_persample_norm=client_kwargs['use_persample_norm'],
        use_clip=client_kwargs['use_clip'],
        use_gpi=client_kwargs['use_gpi'],
    ).to(device)
    server_model = DefendedServerModel(
        base_server,
        use_grad_defense=server_kwargs['use_grad_defense'],
        use_server_sanitization=server_kwargs['use_server_sanitization'],
    ).to(device)
    return client_model, server_model


def train_or_load(config_name, client_model, server_model, train_loader, test_loader, device):
    _, server_kwargs, _ = ABLATION_CONFIGS[config_name]
    ckpt_path = f"{Config.SAVE_DIR}/ablation_{config_name}_{Config.MODEL_NAME.lower()}_{Config.DATASET}.pth"

    if os.path.exists(ckpt_path):
        print(f"\n[✓] Found checkpoint for '{config_name}': {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        client_model.load_state_dict(ckpt.get('client_state_dict', ckpt.get('client_state', {})), strict=False)
        server_model.load_state_dict(ckpt.get('server_state_dict', ckpt.get('server_state', {})), strict=False)
        return

    print(f"\n[!] No checkpoint for '{config_name}'. Training...")

    if config_name == 'baseline':
        from all_split_learning.split_learning import SplitLearningTrainer
        trainer = SplitLearningTrainer(client_model=client_model, server_model=server_model,
                                       train_loader=train_loader, test_loader=test_loader)
        trainer.train()
        torch.save({'client_state': client_model.state_dict(),
                   'server_state': server_model.state_dict()}, ckpt_path)
        return

    criterion = torch.nn.CrossEntropyLoss()
    client_optimizer = torch.optim.Adam(client_model.parameters(), lr=Config.LEARNING_RATE)
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=Config.LEARNING_RATE)

    trainer = DefendedSplitLearningTrainer(
        client_model=client_model, server_model=server_model,
        train_loader=train_loader, val_loader=test_loader,
        client_optimizer=client_optimizer, server_optimizer=server_optimizer,
        criterion=criterion, device=device,
        checkpoint_dir=Config.SAVE_DIR, results_dir=Config.RESULTS_DIR,
        decorr_weight=server_kwargs['decorr_weight'], dcor_weight=server_kwargs['dcor_weight'],
    )
    trainer.train()

    torch.save({'client_state_dict': client_model.state_dict(),
               'server_state_dict': server_model.state_dict()}, ckpt_path)
    print(f"  Saved ablation checkpoint → {ckpt_path}")



MAX_IMAGES  = 32
ITERATIONS  = 1000
HIJACK_EPOCHS    = 5
CRITIC_ITERS     = 5
LEAKAGE_EPOCHS   = 5
POSITIVE_CLASS   = 0
POSITIVE_RATIO   = 0.1
LEAKAGE_BATCH    = 128
LEAKAGE_LR       = 1e-4
WARMUP_EPOCHS    = 5
INFERENCE_EPOCHS = 5
INJECTION_EPOCHS = 10
VILLAIN_BATCH    = 128
TARGET_LABEL     = 0
TRIGGER_BETA     = 1.0
TRIGGER_FRACTION = 0.5
DROPOUT_KEEP     = 0.75
GAMMA_LOW        = 0.6
GAMMA_HIGH       = 1.2
POISON_RATE      = 0.01
CANDIDATES       = 14
BACKDOOR_TARGET_LABEL     = 0
BACKDOOR_POISON_RATE      = 0.05
BACKDOOR_PATCH_SIZE       = 4
BACKDOOR_TRIGGER_VALUE    = 1.0
BACKDOOR_TRAIN_EPOCHS     = 10
BACKDOOR_SURROGATE_EPOCHS = 5


def run_whitebox(client_model, test_loader, device, mean, std, tag):
    attacker = WhiteBoxInversionAttack(client_model=client_model, dataset=Config.DATASET, iterations=ITERATIONS, lr=1e-2)
    tracker = AttackMetricsTracker()
    images_processed = 0
    for inputs, _ in test_loader:
        if images_processed >= MAX_IMAGES:
            break
        inputs = inputs.to(device)[:MAX_IMAGES - images_processed]
        with torch.no_grad():
            target_smashed = client_model(inputs)
        reconstructed = attacker.reconstruct(target_smashed, inputs.shape)
        inputs_denorm = torch.clamp(inputs * std + mean, 0, 1)
        tracker.log_batch(inputs_denorm, reconstructed)
        images_processed += inputs.shape[0]
    summary = tracker.get_summary()
    print(f"  [WhiteBox/{tag}] PSNR: {summary['mean_psnr']:.2f} dB | SSIM: {summary['mean_ssim']:.4f}")
    return summary


def run_unsplit(client_model, build_fresh_client, in_channels, test_loader, tag):
    attacker = UnSplitAttack(
        client_model=client_model, in_channels=in_channels, clone_builder=build_fresh_client,
        main_iters=Config.unsplit_main_iters, input_iters=Config.unsplit_input_iters, model_iters=Config.unsplit_model_iters
    )
    summary = attacker.run_attack(test_loader, num_batches=MAX_IMAGES // 32 or 1)
    print(f"  [UnSplit/{tag}] PSNR: {summary['psnr']:.2f} dB | SSIM: {summary['ssim']:.4f}")
    return summary


def run_ae(client_model, train_loader, test_loader, device, tag):
    summary, _, _ = run_ae_decoder_attack(
        client_model=client_model, train_loader=train_loader, test_loader=test_loader, device=device,
        dataset=Config.DATASET, preprocess_fn=None, ae_epochs=50, label=f'{Config.MODEL_NAME} AE ({tag})'
    )
    print(f"  [AE_Decoder/{tag}] PSNR: {summary['mean_psnr']:.2f} dB | SSIM: {summary['mean_ssim']:.4f}")
    return summary


def run_fsha(client_model, build_fresh_client, in_channels, train_loader, test_loader, tag):
    attacker = FSHAAttack(client_model=client_model, in_channels=in_channels, dataset=Config.DATASET,
                          pilot_builder=build_fresh_client, critic_iters=CRITIC_ITERS)
    attacker.hijack(train_loader, test_loader, epochs=HIJACK_EPOCHS)
    summary = attacker.reconstruct(train_loader, num_images=MAX_IMAGES)
    print(f"  [FSHA/{tag}] PSNR: {summary['psnr']:.2f} dB | SSIM: {summary['ssim']:.4f}")
    return summary


def run_label_leakage(build_fresh_split, train_loader, tag):
    binary_train_loader, binary_test_loader = build_binary_split_loaders(
        train_loader.dataset, target_class=POSITIVE_CLASS, positive_ratio=POSITIVE_RATIO, batch_size=LEAKAGE_BATCH
    )
    leakage_client, leakage_server = build_fresh_split(num_classes=1)
    attacker = GradientNormLabelLeakageAttack(
        client_model=leakage_client, server_model=leakage_server, dataset=Config.DATASET,
        target_class=POSITIVE_CLASS, learning_rate=LEAKAGE_LR
    )
    summary = attacker.run(binary_train_loader, binary_test_loader, epochs=LEAKAGE_EPOCHS)
    print(f"  [LabelLeakage/{tag}] Norm AUC: {summary['q95_norm_leak_auc_cut']:.4f}")
    return summary


def run_villain(build_fresh_split, train_loader, test_loader, tag):
    indexed_loader = build_indexed_loader(train_loader.dataset, batch_size=VILLAIN_BATCH, shuffle=True)
    villain_client, villain_server = build_fresh_split(num_classes=Config.NUM_CLASSES)
    attacker = VILLAINBackdoorAttack(
        client_model=villain_client, server_model=villain_server, base_dataset=train_loader.dataset,
        dataset=Config.DATASET, num_classes=Config.NUM_CLASSES, target_label=TARGET_LABEL, beta=TRIGGER_BETA,
        trigger_fraction=TRIGGER_FRACTION, dropout_keep=DROPOUT_KEEP, gamma_low=GAMMA_LOW, gamma_high=GAMMA_HIGH,
        poison_rate=POISON_RATE, candidates_per_batch=CANDIDATES
    )
    attacker.warmup(indexed_loader, epochs=WARMUP_EPOCHS)
    baseline, _ = attacker.evaluate(test_loader)
    attacker.infer_labels(indexed_loader, epochs=INFERENCE_EPOCHS)
    attacker.fabricate_trigger(indexed_loader)
    attacker.inject_backdoor(indexed_loader, test_loader, epochs=INJECTION_EPOCHS)
    summary = attacker.summarise(test_loader=test_loader, clean_baseline=baseline)
    print(f"  [VILLAIN/{tag}] LIA: {summary['lia']:.2f}% | ASR: {summary['asr']:.2f}% | CDA: {summary['cda']:.2f}%")
    return summary


def run_backdoor(mode, build_fresh_split, build_fresh_client, train_loader, test_loader, tag):
    b_client, b_server = build_fresh_split(num_classes=Config.NUM_CLASSES)
    kwargs = dict(
        client_model=b_client, server_model=b_server, base_dataset=train_loader.dataset,
        dataset=Config.DATASET, num_classes=Config.NUM_CLASSES, mode=mode, target_label=BACKDOOR_TARGET_LABEL,
        poison_rate=BACKDOOR_POISON_RATE, patch_size=BACKDOOR_PATCH_SIZE, trigger_value=BACKDOOR_TRIGGER_VALUE,
        model_tag=f"ablation_{tag}_{Config.MODEL_NAME.lower()}_sl",
    )
    if mode == 'server':
        kwargs['surrogate_builder'] = build_fresh_client
    attacker = BackdoorPoisonAttack(**kwargs)
    if mode == 'server':
        attacker.pretrain_server_backdoor(test_loader, epochs=BACKDOOR_SURROGATE_EPOCHS)
    attacker.train(train_loader, test_loader, epochs=BACKDOOR_TRAIN_EPOCHS)
    summary = attacker.summarise()
    print(f"  [Backdoor_{mode}/{tag}] ASR: {summary['asr']:.2f}% | CDA: {summary['cda']:.2f}%")
    return summary


ATTACK_RUNNERS = {
    'whitebox':        lambda ctx: run_whitebox(ctx['client_model'], ctx['test_loader'], ctx['device'], ctx['mean'], ctx['std'], ctx['tag']),
    'unsplit':         lambda ctx: run_unsplit(ctx['client_model'], ctx['build_fresh_client'], ctx['in_channels'], ctx['test_loader'], ctx['tag']),
    'ae_decoder':      lambda ctx: run_ae(ctx['client_model'], ctx['train_loader'], ctx['test_loader'], ctx['device'], ctx['tag']),
    'fsha':            lambda ctx: run_fsha(ctx['client_model'], ctx['build_fresh_client'], ctx['in_channels'], ctx['train_loader'], ctx['test_loader'], ctx['tag']),
    'label_leakage':   lambda ctx: run_label_leakage(ctx['build_fresh_split'], ctx['train_loader'], ctx['tag']),
    'villain':         lambda ctx: run_villain(ctx['build_fresh_split'], ctx['train_loader'], ctx['test_loader'], ctx['tag']),
    'backdoor_client': lambda ctx: run_backdoor('client', ctx['build_fresh_split'], ctx['build_fresh_client'], ctx['train_loader'], ctx['test_loader'], ctx['tag']),
    'backdoor_server': lambda ctx: run_backdoor('server', ctx['build_fresh_split'], ctx['build_fresh_client'], ctx['train_loader'], ctx['test_loader'], ctx['tag']),
}


if __name__ == "__main__":

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")
    print(f"Model: {Config.MODEL_NAME} | Dataset: {Config.DATASET} | Cut layer: {Config.CUT_LAYER}")

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()

    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    image_size  = 28 if Config.DATASET == 'MNIST' else 32

    if Config.DATASET == 'CIFAR10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(device)
        std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(device)
    else:
        mean = torch.tensor([0.1307]).view(1,1,1,1).to(device)
        std  = torch.tensor([0.3081]).view(1,1,1,1).to(device)

    all_rows = []

    for config_name, (client_kwargs, server_kwargs, attack_keys) in ABLATION_CONFIGS.items():
        print("\n" + "="*70)
        print(f"  ABLATION CONFIG: {config_name}  (attacks: {attack_keys})")
        print("="*70)

       
        client_model, server_model = build_config_models(config_name, device, in_channels, image_size)
        train_or_load(config_name, client_model, server_model, train_loader, test_loader, device)
        client_model.eval()
        server_model.eval()

        def build_fresh_client(_config_name=config_name):
            base_client, _ = build_base_client_server(device, in_channels)
            client_kwargs_, _, _ = ABLATION_CONFIGS[_config_name]
            if client_kwargs_ is None:
                return base_client.to(device)
            return DefendedClientModel(base_client, in_channels, image_size, device, **client_kwargs_).to(device)

        def build_fresh_split(num_classes, _config_name=config_name):
            c, s = build_config_models(_config_name, device, in_channels, image_size, num_classes=num_classes)
            ckpt_path = f"{Config.SAVE_DIR}/ablation_{_config_name}_{Config.MODEL_NAME.lower()}_{Config.DATASET}.pth"
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                c.load_state_dict(ckpt.get('client_state_dict', ckpt.get('client_state', {})), strict=False)
                if num_classes == Config.NUM_CLASSES:
                    s.load_state_dict(ckpt.get('server_state_dict', ckpt.get('server_state', {})), strict=False)
            return c, s

        ctx = {
            'client_model': client_model, 'server_model': server_model,
            'build_fresh_client': build_fresh_client, 'build_fresh_split': build_fresh_split,
            'train_loader': train_loader, 'test_loader': test_loader,
            'device': device, 'mean': mean, 'std': std, 'in_channels': in_channels,
            'tag': config_name,
        }

        for attack_key in attack_keys:
            summary = ATTACK_RUNNERS[attack_key](ctx)
            row = {'config': config_name, 'attack': attack_key}
            row.update(summary)
            all_rows.append(row)

    combined_path = f"{Config.RESULTS_DIR}/ablation_results_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv"
    pd.DataFrame(all_rows).to_csv(combined_path, index=False)
    print(f"\n  Ablation results saved → {combined_path}")

    print("\n" + "="*70)
    print("  ABLATION SUMMARY (component contribution per targeted attack)")
    print("="*70)
    df = pd.DataFrame(all_rows)
    for attack_key in ALL_ATTACKS:
        subset = df[df['attack'] == attack_key]
        if len(subset) <= 1:
            continue
        print(f"\n  -- {attack_key} --")
        print(subset.set_index('config').to_string())