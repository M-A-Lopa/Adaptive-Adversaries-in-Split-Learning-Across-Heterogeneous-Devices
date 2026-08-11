import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import Config
from dataset import DatasetLoader
from all_model.models import ClientModel, ServerModel
from all_model.ressfl_models import build_ae, custom_AE
from all_split_learning.ressfl_split_learning import ResSFLTrainer
from all_defences.ressfl_defense import WindowedSSIM, denormalize
from all_attacks.attacks_whitebox import WhiteBoxInversionAttack, AttackMetricsTracker
from all_attacks.ae_decoder_attack import run_ae_decoder_attack, save_ae_attack_visualization
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

MAX_IMAGES_ATTACK   = 32   
ATTACK_ITERATIONS   = 1000 
MODEL_BASED_EPOCHS  = 50  


def load_or_train_ressfl(device, train_loader, test_loader):
    in_channels  = 1 if Config.DATASET == 'MNIST' else 3
    client_model = ClientModel(in_channels=in_channels).to(device)
    server_model = ServerModel(num_classes=Config.NUM_CLASSES).to(device)

    ressfl_ckpt  = f"{Config.SAVE_DIR}/best_ressfl_{Config.DATASET}.pth"
    vanilla_ckpt = f"{Config.SAVE_DIR}/best_vanilla_sl_{Config.DATASET}.pth"

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
        print("    Initializing client+server from vanilla weights...")
        vanilla = torch.load(vanilla_ckpt, map_location=device)
        client_model.load_state_dict(vanilla['client_state'])
        server_model.load_state_dict(vanilla['server_state'])
        print(f"    Vanilla accuracy: {vanilla['best_acc']:.2f}%")
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


def run_ressfl_attack(attack_type, client_model, train_loader, test_loader, device):
    if Config.DATASET == 'CIFAR10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465],
                            device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.2023, 0.1994, 0.2010],
                            device=device).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.1307], device=device).view(1, 1, 1, 1)
        std  = torch.tensor([0.3081], device=device).view(1, 1, 1, 1)

    if attack_type == 'whitebox':
        print("\n" + "="*60)
        print("  WHITE-BOX ATTACK AGAINST RESSFL DEFENSE")
        print("="*60)

        attacker = WhiteBoxInversionAttack(client_model=client_model, dataset=Config.DATASET, iterations=ATTACK_ITERATIONS, lr=1e-2, use_normalization=True)
        tracker = AttackMetricsTracker()

        images_processed  = 0
        originals_vis     = []
        reconstructed_vis = []

        client_model.eval()

        for inputs, _ in test_loader:
            if images_processed >= MAX_IMAGES_ATTACK:
                break
            inputs    = inputs.to(device)
            remaining = MAX_IMAGES_ATTACK - images_processed
            inputs    = inputs[:remaining]

            print(f"\n  Inverting {inputs.shape[0]} image(s) "
                  f"[{images_processed + inputs.shape[0]}"
                  f"/{MAX_IMAGES_ATTACK} total]...")

            originals_dn = torch.clamp(inputs * std + mean, 0, 1)

            with torch.no_grad():
                target_smashed = client_model(inputs)

            reconstructed = attacker.reconstruct(target_smashed, inputs.shape)

            tracker.log_batch(originals_dn, reconstructed)
            images_processed += inputs.shape[0]

            if len(originals_vis) == 0:
                originals_vis     = originals_dn[:8].cpu()
                reconstructed_vis = reconstructed[:8].cpu()

        summary = tracker.get_summary()

        print("\n" + "="*60)
        print("  WHITE-BOX ATTACK vs RESSFL — RESULTS")
        print("="*60)
        print(f"  PSNR : {summary['mean_psnr']:.2f} dB")
        print(f"  SSIM : {summary['mean_ssim']:.4f}")
        print(f"  Images: {images_processed}")
        print("="*60)

        return summary, originals_vis, reconstructed_vis

    elif attack_type == 'ae_decoder':
        return run_ae_decoder_attack(client_model=client_model, train_loader=train_loader, test_loader=test_loader, device=device, dataset=Config.DATASET, preprocess_fn=None, ae_epochs=MODEL_BASED_EPOCHS, label='ResSFL AE Decoder')

    else:
        raise ValueError(
            f"Unknown attack_type '{attack_type}'. "
            f"Use 'whitebox' or 'ae_decoder'.")


def save_ressfl_visualization(originals, reconstructed, baseline_summary, ressfl_summary, attack_type):
    num = min(8, len(originals))
    fig = plt.figure(figsize=(num * 2, 5))
    gs  = gridspec.GridSpec(2, num, hspace=0.35)

    for i in range(num):
        ax1 = fig.add_subplot(gs[0, i])
        img = originals[i].permute(1, 2, 0).numpy()
        if img.shape[2] == 1:
            ax1.imshow(img.squeeze(2), cmap='gray')
        else:
            ax1.imshow(np.clip(img, 0, 1))
        ax1.axis('off')
        if i == 0:
            ax1.set_title('Original', fontsize=10, fontweight='bold')

        ax2 = fig.add_subplot(gs[1, i])
        rec = reconstructed[i].permute(1, 2, 0).numpy()
        if rec.shape[2] == 1:
            ax2.imshow(rec.squeeze(2), cmap='gray')
        else:
            ax2.imshow(np.clip(rec, 0, 1))
        ax2.axis('off')
        if i == 0:
            ax2.set_title('Reconstructed\n(Attack+ResSFL)',
                          fontsize=10, fontweight='bold')

    plt.suptitle(
        f'{attack_type} Attack vs ResSFL Defense — {Config.DATASET}\n'
        f'Baseline: PSNR={baseline_summary["mean_psnr"]:.2f}dB '
        f'SSIM={baseline_summary["mean_ssim"]:.4f} | '
        f'ResSFL: PSNR={ressfl_summary["mean_psnr"]:.2f}dB '
        f'SSIM={ressfl_summary["mean_ssim"]:.4f}',
        fontsize=10, fontweight='bold')

    tag  = attack_type.lower().replace('-', '').replace(' ', '_')
    path = f"{Config.RESULTS_DIR}/ressfl_{tag}_comparison_{Config.DATASET}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Visualization saved → {path}")


def print_full_comparison_table(baseline, pgsl, ressfl):
    print("\n" + "="*68)
    print("   FULL DEFENSE COMPARISON TABLE")
    print("="*68)
    print(f"{'Method':<25} {'PSNR (dB)':>12} {'SSIM':>10} {'Note'}")
    print("-"*68)
    print(f"  {'No Defense (baseline)':<23} "
          f"{baseline['mean_psnr']:>12.2f} "
          f"{baseline['mean_ssim']:>10.4f}  attack confirmed")

    if pgsl is not None:
        psnr_d = baseline['mean_psnr'] - pgsl['mean_psnr']
        ssim_d = baseline['mean_ssim'] - pgsl['mean_ssim']
        tag    = "✓ below 0.3" if pgsl['mean_ssim'] < 0.3 else "partial defense"
        print(f"  {'PGSL Defense':<23} "
              f"{pgsl['mean_psnr']:>12.2f} "
              f"{pgsl['mean_ssim']:>10.4f}  {tag}")

    psnr_d = baseline['mean_psnr'] - ressfl['mean_psnr']
    ssim_d = baseline['mean_ssim'] - ressfl['mean_ssim']
    tag    = "✓ below 0.3" if ressfl['mean_ssim'] < 0.3 else "partial defense"
    print(f"  {'ResSFL Defense':<23} "
          f"{ressfl['mean_psnr']:>12.2f} "
          f"{ressfl['mean_ssim']:>10.4f}  {tag}")

    print("-"*68)
    print(f"  Threshold (Section 3.3): SSIM = 0.3")
    print(f"  Below 0.3 = reconstruction unrecognizable")
    print("="*68)

if __name__ == "__main__":

    print("="*60)
    print("  RESSFL DEFENSE EXPERIMENT")
    print(f"  Dataset : {Config.DATASET}")
    print("="*60)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"  Device  : {device}\n")

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    dataset      = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()

    client_model, server_model = load_or_train_ressfl(device, train_loader, test_loader)

    baseline_csv = f"{Config.RESULTS_DIR}/attack_evaluation_{Config.DATASET}.csv"
    if os.path.exists(baseline_csv):
        baseline_df      = pd.read_csv(baseline_csv)
        baseline_summary = baseline_df.iloc[0].to_dict()
        print(f"\n[✓] Loaded baseline from: {baseline_csv}")
        print(f"    Baseline PSNR: {baseline_summary['mean_psnr']:.2f} dB")
        print(f"    Baseline SSIM: {baseline_summary['mean_ssim']:.4f}")
    else:
        print("\n[!] No baseline CSV found.")
        print("    Run main.py with RUN_ATTACK=True first.")
        baseline_summary = {'mean_psnr': 0.0, 'mean_ssim': 0.0}

    pgsl_csv = f"{Config.RESULTS_DIR}/pgsl_attack_evaluation_{Config.DATASET}.csv"
    pgsl_summary = None
    if os.path.exists(pgsl_csv):
        pgsl_df      = pd.read_csv(pgsl_csv)
        pgsl_summary = pgsl_df.iloc[0].to_dict()
        print(f"\n[✓] Loaded PGSL results from: {pgsl_csv}")

    print("\n  Select attack to run against ResSFL:")
    print("  1 = White-Box rMSE attack")
    print("  2 = AE Decoder attack (ResSFL paper protocol)")
    print("  3 = Both")
    attack_choice = input("  Choice (1/2/3, default 2): ").strip()

    run_whitebox = attack_choice in ('1', '3')
    run_ae_dec   = attack_choice in ('', '2', '3')

    wb_summary = None
    if run_whitebox:
        wb_summary, wb_orig, wb_recon = run_ressfl_attack( attack_type='whitebox', client_model=client_model, train_loader=train_loader, test_loader=test_loader, device=device)
        pd.DataFrame([wb_summary]).to_csv(
            f"{Config.RESULTS_DIR}/ressfl_wb_attack_{Config.DATASET}.csv", index=False)
        print(f"  Results saved → "
              f"{Config.RESULTS_DIR}/ressfl_wb_attack_{Config.DATASET}.csv")

        if len(wb_orig) > 0:
            save_ressfl_visualization( wb_orig, wb_recon, baseline_summary, wb_summary, attack_type='White-Box')

    ae_summary = None
    if run_ae_dec:
        ae_summary, ae_orig, ae_recon = run_ressfl_attack(attack_type='ae_decoder', client_model=client_model, train_loader=train_loader, test_loader=test_loader, device=device)
        pd.DataFrame([ae_summary]).to_csv(
            f"{Config.RESULTS_DIR}/ressfl_ae_attack_{Config.DATASET}.csv",
            index=False)
        print(f"  Results saved → "
              f"{Config.RESULTS_DIR}/ressfl_ae_attack_{Config.DATASET}.csv")

        if len(ae_orig) > 0:
            save_ae_attack_visualization(originals=ae_orig, reconstructed=ae_recon, baseline_summary=baseline_summary, defense_summary=ae_summary, defense_name='ResSFL', dataset=Config.DATASET, results_dir=Config.RESULTS_DIR)

    if wb_summary is not None and ae_summary is not None:
        print("\n" + "="*68)
        print("   RESSFL — WHITE-BOX vs AE DECODER COMPARISON")
        print("="*68)
        print(f"{'Attack':<30} {'PSNR (dB)':>12} {'SSIM':>10}")
        print("-"*55)
        print(f"  {'No Defense (baseline)':<28} "
              f"{baseline_summary['mean_psnr']:>12.2f} "
              f"{baseline_summary['mean_ssim']:>10.4f}")
        print(f"  {'ResSFL (White-Box)':<28} "
              f"{wb_summary['mean_psnr']:>12.2f} "
              f"{wb_summary['mean_ssim']:>10.4f}")
        print(f"  {'ResSFL (AE Decoder)':<28} "
              f"{ae_summary['mean_psnr']:>12.2f} "
              f"{ae_summary['mean_ssim']:>10.4f}")
        print("="*68)

    final_ressfl = ae_summary if ae_summary is not None else wb_summary
    if final_ressfl is not None:
        print_full_comparison_table(
            baseline_summary, pgsl_summary, final_ressfl)

    print("\n  ResSFL experiment complete.")
    print(f"  All outputs in: {Config.RESULTS_DIR}/")