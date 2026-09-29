import matplotlib
matplotlib.use('Agg')   # file-only backend -- avoids the Tk crash entirely

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from config import Config
from dataset import DatasetLoader
from all_model.pgsl_models import build_pgsl_client, PGSLServerModel
from all_defences.pgsl_defense import PGSLDefenseModules, AdaptiveWeightedDecisionFusion
from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack


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


def load_pgsl_checkpoint_only(device, in_channels, image_size):
    """
    Loads the ALREADY-TRAINED PGSL checkpoint. Does NOT train --
    training already completed in the previous run, this just pulls
    the saved weights back in.
    """
    base_client = build_pgsl_client(original_in_channels=in_channels).to(device)
    smashed_channels = _smashed_channels_for(base_client, in_channels, image_size, device)
    base_server = PGSLServerModel(num_classes=Config.NUM_CLASSES, smashed_channels=smashed_channels).to(device)

    pgsl_ckpt_path = f"{Config.SAVE_DIR}/best_pgsl_{Config.MODEL_NAME.lower()}_sl_{Config.DATASET}.pth"
    if not os.path.exists(pgsl_ckpt_path):
        raise FileNotFoundError(
            f"No PGSL checkpoint found at {pgsl_ckpt_path}. "
            f"Cannot recover -- original training must be re-run."
        )

    ckpt = torch.load(pgsl_ckpt_path, map_location=device)
    base_client.load_state_dict(ckpt['client_state'])
    base_server.load_state_dict(ckpt['server_state'])
    print(f"[✓] Loaded PGSL checkpoint: {pgsl_ckpt_path}")
    print(f"    Best PGSL accuracy: {ckpt['best_acc']:.2f}%")

    return base_client, base_server, smashed_channels


def load_csv_summary(path, required_keys):
    """
    Pulls a previously-saved single-row attack CSV back into a dict,
    exactly matching the structure the original run produced in memory.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected result file not found: {path}\n"
            f"This attack's CSV should already exist from the earlier run."
        )
    row = pd.read_csv(path).iloc[0]
    return {k: row[k] for k in required_keys}


if __name__ == "__main__":

    print("=" * 60)
    print("  PGSL RECOVERY RUN")
    print(f"  Model   : {Config.MODEL_NAME}")
    print(f"  Dataset : {Config.DATASET}")
    print("=" * 60)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"  Device  : {device}\n")

    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    image_size  = 28 if Config.DATASET == 'MNIST' else 32

    # ── Step 1: pull trained weights straight from checkpoint ──────────────
    base_client, base_server, smashed_channels = load_pgsl_checkpoint_only(
        device, in_channels, image_size
    )

    def build_fresh_client():
        base = build_pgsl_client(original_in_channels=in_channels).to(device)
        return PGSLClientAdapter(base).to(device)

    def build_fresh_split(num_classes):
        b_client = build_pgsl_client(original_in_channels=in_channels).to(device)
        b_server = PGSLServerModel(num_classes=num_classes, smashed_channels=smashed_channels).to(device)
        return PGSLClientAdapter(b_client).to(device), PGSLServerAdapter(b_server).to(device)

    # ── Step 2: pull the 7 already-completed attack results from disk ──────
    tag = f"{Config.MODEL_NAME.lower()}_{Config.DATASET}"

    print("\n[✓] Loading previously-saved attack results from disk...")

    wb_summary = load_csv_summary(
        f"{Config.RESULTS_DIR}/pgsl_attack_whitebox_{tag}.csv",
        ['mean_psnr', 'mean_ssim']
    )
    unsplit_summary = load_csv_summary(
        f"{Config.RESULTS_DIR}/pgsl_attack_unsplit_{tag}.csv",
        ['psnr', 'ssim']
    )
    ae_summary = load_csv_summary(
        f"{Config.RESULTS_DIR}/pgsl_attack_ae_decoder_{tag}.csv",
        ['mean_psnr', 'mean_ssim']
    )
    fsha_summary = load_csv_summary(
        f"{Config.RESULTS_DIR}/pgsl_attack_fsha_{tag}.csv",
        ['psnr', 'ssim', 'mse']
    )
    leakage_summary = load_csv_summary(
        f"{Config.RESULTS_DIR}/pgsl_attack_label_leakage_{tag}.csv",
        ['q95_norm_leak_auc_cut', 'q95_majority_accuracy_cut', 'test_auc']
    )
    villain_summary = load_csv_summary(
        f"{Config.RESULTS_DIR}/pgsl_attack_villain_{tag}.csv",
        ['lia', 'asr', 'cda']
    )
    backdoor_client_summary = load_csv_summary(
        f"{Config.RESULTS_DIR}/pgsl_attack_backdoor_poison_client_{tag}.csv",
        ['asr', 'cda']
    )

    print("[✓] All 7 prior attack results loaded successfully.\n")

    # ── Step 3: re-run ONLY the attack that crashed -- Backdoor Server ──────
    print("=" * 60)
    print(f"  BACKDOOR POISONING (SERVER-SIDE) vs PGSL — {Config.MODEL_NAME}")
    print("=" * 60)

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()

    BACKDOOR_TARGET_LABEL     = 0
    BACKDOOR_POISON_RATE      = 0.05
    BACKDOOR_PATCH_SIZE       = 4
    BACKDOOR_TRIGGER_VALUE    = 1.0
    BACKDOOR_TRAIN_EPOCHS     = 10
    BACKDOOR_SURROGATE_EPOCHS = 5

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
    backdoor_server_attacker.save_visualization(tag=f"pgsl_{tag}")

    pd.DataFrame([backdoor_server_summary]).to_csv(
        f"{Config.RESULTS_DIR}/pgsl_attack_backdoor_poison_server_{tag}.csv", index=False)
    pd.DataFrame(backdoor_server_attacker.history).to_csv(
        f"{Config.RESULTS_DIR}/pgsl_backdoor_poison_server_epochs_{tag}.csv", index=False)

    print("[✓] Backdoor Server attack complete and saved.\n")

    blank = "—"
    print("\n" + "=" * 94)
    print(f"   ALL ATTACKS vs PGSL SUMMARY — {Config.MODEL_NAME} on {Config.DATASET}")
    print("=" * 94)
    print(f"{'Attack':<20} {'PSNR (dB)':>11} {'SSIM':>9} {'MSE':>9} {'Leak AUC':>10} {'LIA (%)':>9} {'ASR (%)':>9} {'CDA (%)':>9}")
    print("-" * 94)
    print(f"  {'White-Box':<18} {wb_summary['mean_psnr']:>11.2f} {wb_summary['mean_ssim']:>9.4f} {blank:>9} {blank:>10} {blank:>9} {blank:>9} {blank:>9}")
    print(f"  {'UnSplit':<18} {unsplit_summary['psnr']:>11.2f} {unsplit_summary['ssim']:>9.4f} {blank:>9} {blank:>10} {blank:>9} {blank:>9} {blank:>9}")
    print(f"  {'AE Decoder':<18} {ae_summary['mean_psnr']:>11.2f} {ae_summary['mean_ssim']:>9.4f} {blank:>9} {blank:>10} {blank:>9} {blank:>9} {blank:>9}")
    print(f"  {'FSHA':<18} {fsha_summary['psnr']:>11.2f} {fsha_summary['ssim']:>9.4f} {fsha_summary['mse']:>9.5f} {blank:>10} {blank:>9} {blank:>9} {blank:>9}")
    print(f"  {'Label Leakage':<18} {blank:>11} {blank:>9} {blank:>9} {leakage_summary['q95_norm_leak_auc_cut']:>10.4f} {blank:>9} {blank:>9} {leakage_summary['test_auc']:>9.4f}")
    print(f"  {'VILLAIN':<18} {blank:>11} {blank:>9} {blank:>9} {blank:>10} {villain_summary['lia']:>9.2f} {villain_summary['asr']:>9.2f} {villain_summary['cda']:>9.2f}")
    print(f"  {'Backdoor(Client)':<18} {blank:>11} {blank:>9} {blank:>9} {blank:>10} {blank:>9} {backdoor_client_summary['asr']:>9.2f} {backdoor_client_summary['cda']:>9.2f}")
    print(f"  {'Backdoor(Server)':<18} {blank:>11} {blank:>9} {blank:>9} {blank:>10} {blank:>9} {backdoor_server_summary['asr']:>9.2f} {backdoor_server_summary['cda']:>9.2f}")
    print("=" * 94)

    combined_path = f"{Config.RESULTS_DIR}/pgsl_all_attacks_{tag}.csv"
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