import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import Config
from dataset import DatasetLoader
from all_model.pgsl_models import PGSLClientModel, PGSLServerModel
from all_split_learning.pgsl_split_learning import PGSLSplitLearningTrainer
from all_defences.pgsl_defense import PGSLDefenseModules, AdaptiveWeightedDecisionFusion
from all_attacks.attacks_whitebox import WhiteBoxInversionAttack, AttackMetricsTracker
from all_attacks.ae_decoder_attack import run_ae_decoder_attack, save_ae_attack_visualization

MAX_IMAGES_ATTACK = 32  
ATTACK_ITERATIONS = 1000 

def pgsl_preprocess_fn(inputs):
    if Config.DATASET == 'CIFAR10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465],
                            device=inputs.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.2023, 0.1994, 0.2010],
                            device=inputs.device).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.1307],
                            device=inputs.device).view(1, 1, 1, 1)
        std  = torch.tensor([0.3081],
                            device=inputs.device).view(1, 1, 1, 1)
    denorm = torch.clamp(inputs * std + mean, 0.0, 1.0)
    
    return PGSLDefenseModules.space_to_depth_downsample(denorm, saliency_map=None)
        
def adapt_vanilla_weights_to_pgsl(vanilla_ckpt, client_model, server_model, device):
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


def load_or_train_pgsl(device, train_loader, test_loader):
    in_channels  = 1 if Config.DATASET == 'MNIST' else 3
    client_model = PGSLClientModel(original_in_channels=in_channels).to(device)
    server_model = PGSLServerModel(num_classes=Config.NUM_CLASSES).to(device)

    pgsl_ckpt_path    = f"{Config.SAVE_DIR}/best_pgsl_sl_{Config.DATASET}.pth"
    vanilla_ckpt_path = f"{Config.SAVE_DIR}/best_vanilla_sl_{Config.DATASET}.pth"

    if os.path.exists(pgsl_ckpt_path):
        print(f"\n[✓] Found PGSL checkpoint: {pgsl_ckpt_path}")
        ckpt = torch.load(pgsl_ckpt_path, map_location=device)
        client_model.load_state_dict(ckpt['client_state'])
        server_model.load_state_dict(ckpt['server_state'])
        print(f"    Best PGSL accuracy: {ckpt['best_acc']:.2f}%")

    elif os.path.exists(vanilla_ckpt_path):
        print(f"\n[!] No PGSL checkpoint found.")
        print(f"[✓] Found vanilla checkpoint: {vanilla_ckpt_path}")
        print("    Adapting vanilla weights to PGSL architecture...")
        vanilla_ckpt = torch.load(vanilla_ckpt_path, map_location=device)
        adapt_vanilla_weights_to_pgsl(vanilla_ckpt, client_model, server_model, device)
        print("    Starting PGSL training from adapted weights...")

        trainer = PGSLSplitLearningTrainer( client_model=client_model, server_model=server_model, train_loader=train_loader, test_loader=test_loader)
        trainer.train()
        trainer.save_results()

    else:
        print(f"\n[!] No checkpoint found. Training PGSL from scratch...")
        trainer = PGSLSplitLearningTrainer(client_model=client_model, server_model=server_model, train_loader=train_loader, test_loader=test_loader)
        trainer.train()
        trainer.save_results()

    return client_model, server_model


def run_pgsl_attack(attack_type, client_model, train_loader, test_loader, device):
    in_channels = 1 if Config.DATASET == 'MNIST' else 3

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
        print("  WHITE-BOX ATTACK AGAINST PGSL DEFENSE")
        print("="*60)

        attacker = WhiteBoxInversionAttack(client_model=client_model, dataset=Config.DATASET, iterations=ATTACK_ITERATIONS, lr=1e-2, use_normalization=False)
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
                client_tensor  = pgsl_preprocess_fn(inputs)
                target_smashed = client_model(client_tensor)

            reconstructed_pgsl = attacker.reconstruct(target_smashed, client_tensor.shape)
            reconstructed_orig = PGSLDefenseModules.depth_to_space_upsample(reconstructed_pgsl, original_channels=in_channels)

            tracker.log_batch(originals_dn, reconstructed_orig)
            images_processed += inputs.shape[0]

            if len(originals_vis) == 0:
                originals_vis     = originals_dn[:8].cpu()
                reconstructed_vis = reconstructed_orig[:8].cpu()

        summary = tracker.get_summary()

        print("\n" + "="*60)
        print("  WHITE-BOX ATTACK vs PGSL — RESULTS")
        print("="*60)
        print(f"  PSNR : {summary['mean_psnr']:.2f} dB")
        print(f"  SSIM : {summary['mean_ssim']:.4f}")
        print(f"  Images evaluated: {images_processed}")
        print("="*60)

        return summary, originals_vis, reconstructed_vis

    elif attack_type == 'ae_decoder':
        
        return run_ae_decoder_attack(
            client_model=client_model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            dataset=Config.DATASET,
            preprocess_fn=pgsl_preprocess_fn,
            ae_epochs=50,
            label='PGSL AE Decoder'
        )

    else:
        raise ValueError(
            f"Unknown attack_type '{attack_type}'. "
            f"Use 'whitebox' or 'ae_decoder'."
        )


def save_pgsl_visualization(originals, reconstructed, baseline_summary, pgsl_summary):
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
            ax2.set_title('Reconstructed\n(Attack+PGSL)', fontsize=10,
                          fontweight='bold')

    plt.suptitle(
        f'White-Box Attack vs PGSL Defense — {Config.DATASET}\n'
        f'Baseline: PSNR={baseline_summary["mean_psnr"]:.2f}dB '
        f'SSIM={baseline_summary["mean_ssim"]:.4f} | '
        f'PGSL: PSNR={pgsl_summary["mean_psnr"]:.2f}dB '
        f'SSIM={pgsl_summary["mean_ssim"]:.4f}',
        fontsize=11, fontweight='bold')

    save_path = f"{Config.RESULTS_DIR}/pgsl_attack_comparison_{Config.DATASET}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Visualization saved → {save_path}")


def print_comparison_table(baseline_summary, pgsl_summary):
    print("\n" + "="*60)
    print("   THESIS COMPARISON TABLE — PGSL DEFENSE EVALUATION")
    print("="*60)
    print(f"{'Method':<25} {'PSNR (dB)':>12} {'SSIM':>10}")
    print("-"*50)
    print(f"  {'No Defense (baseline)':<23} "
          f"{baseline_summary['mean_psnr']:>12.2f} "
          f"{baseline_summary['mean_ssim']:>10.4f}")
    print(f"  {'PGSL Defense':<23} "
          f"{pgsl_summary['mean_psnr']:>12.2f} "
          f"{pgsl_summary['mean_ssim']:>10.4f}")
    print("-"*50)
    psnr_drop = baseline_summary['mean_psnr'] - pgsl_summary['mean_psnr']
    ssim_drop = baseline_summary['mean_ssim'] - pgsl_summary['mean_ssim']
    print(f"  {'Defense gain':<23} "
          f"{'↓'+str(round(psnr_drop,2))+'dB':>12} "
          f"{'↓'+str(round(ssim_drop,4)):>10}")
    print("="*60)
    print("\n  Interpretation:")
    if pgsl_summary['mean_ssim'] < 0.3:
        print("  PGSL reduced SSIM below 0.3 — attack reconstruction is unrecognizable.")
    else:
        print("  PGSL degraded reconstruction but SSIM above 0.3 — partial recovery still possible.")
    print("  SDANI defense will be evaluated next for comparison.")


if __name__ == "__main__":

    print("="*60)
    print("  PGSL DEFENSE EXPERIMENT")
    print(f"  Dataset : {Config.DATASET}")
    print("="*60)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"  Device  : {device}\n")

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()

    client_model, server_model = load_or_train_pgsl(device, train_loader, test_loader)

    baseline_csv = f"{Config.RESULTS_DIR}/attack_evaluation_{Config.DATASET}.csv"
    if os.path.exists(baseline_csv):
        baseline_df      = pd.read_csv(baseline_csv)
        baseline_summary = baseline_df.iloc[0].to_dict()
        print(f"\n[✓] Loaded baseline results from: {baseline_csv}")
        print(f"    Baseline PSNR: {baseline_summary['mean_psnr']:.2f} dB")
        print(f"    Baseline SSIM: {baseline_summary['mean_ssim']:.4f}")
    else:
        print("\n[!] No baseline CSV found.")
        print("    Run main.py with RUN_ATTACK=True first to get baseline numbers.")
        print("    Using placeholder zeros for comparison.")
        baseline_summary = {'mean_psnr': 0.0, 'mean_ssim': 0.0}

    print("\n  Select attack to run against PGSL:")
    print("  1 = White-Box rMSE attack")
    print("  2 = AE Decoder attack (ResSFL paper protocol)")
    print("  3 = Both")
    attack_choice = input("  Choice (1/2/3, default 1): ").strip()

    run_whitebox  = attack_choice in ('', '1', '3')
    run_ae_dec    = attack_choice in ('2', '3')

    wb_summary = None
    if run_whitebox:
        wb_summary, wb_orig, wb_recon = run_pgsl_attack(attack_type='whitebox', client_model=client_model, train_loader=train_loader, test_loader=test_loader, device=device)
        pd.DataFrame([wb_summary]).to_csv(f"{Config.RESULTS_DIR}/pgsl_attack_evaluation_{Config.DATASET}.csv", index=False)
        print(f"  Results saved → "
              f"{Config.RESULTS_DIR}/pgsl_attack_evaluation_{Config.DATASET}.csv")

        if len(wb_orig) > 0:
            save_pgsl_visualization(wb_orig, wb_recon, baseline_summary, wb_summary)
        print_comparison_table(baseline_summary, wb_summary)

    ae_summary = None
    if run_ae_dec:
        ae_summary, ae_orig, ae_recon = run_pgsl_attack( attack_type='ae_decoder', client_model=client_model, train_loader=train_loader, test_loader=test_loader, device=device)
        pd.DataFrame([ae_summary]).to_csv(
            f"{Config.RESULTS_DIR}/pgsl_ae_attack_{Config.DATASET}.csv", index=False)
        print(f"  Results saved → "
              f"{Config.RESULTS_DIR}/pgsl_ae_attack_{Config.DATASET}.csv")

        if len(ae_orig) > 0:
            save_ae_attack_visualization(originals=ae_orig, reconstructed=ae_recon, baseline_summary=baseline_summary, defense_summary=ae_summary,  defense_name='PGSL', dataset=Config.DATASET, results_dir=Config.RESULTS_DIR)

    if wb_summary is not None and ae_summary is not None:
        print("\n" + "="*68)
        print("   PGSL — WHITE-BOX vs AE DECODER COMPARISON")
        print("="*68)
        print(f"{'Attack':<30} {'PSNR (dB)':>12} {'SSIM':>10}")
        print("-"*55)
        print(f"  {'No Defense (baseline)':<28} "
              f"{baseline_summary['mean_psnr']:>12.2f} "
              f"{baseline_summary['mean_ssim']:>10.4f}")
        print(f"  {'PGSL (White-Box)':<28} "
              f"{wb_summary['mean_psnr']:>12.2f} "
              f"{wb_summary['mean_ssim']:>10.4f}")
        print(f"  {'PGSL (AE Decoder)':<28} "
              f"{ae_summary['mean_psnr']:>12.2f} "
              f"{ae_summary['mean_ssim']:>10.4f}")
        print("="*68)

    print("\n  PGSL experiment complete.")
    print(f"  All outputs in: {Config.RESULTS_DIR}/")
