import os
import torch
import pandas as pd
from config import Config
from dataset import DatasetLoader
from models import ClientModel, ServerModel
from split_learning import SplitLearningTrainer
from all_attacks.attacks_whitebox import WhiteBoxInversionAttack, AttackMetricsTracker
from All_defences.dpsl_defense import DPSLDefense


# ── Settings you can tune ───────────────────────────────────────────────
EPSILON_VALUES = [10, 50, 100, 200, 500, 1000]   # Realistic range for feature size
DELTA          = 1e-5
CLIP_NORM      = 32.0                            # Matches measured MNIST median norm

MAX_IMAGES     = 32                              # Full evaluation run (32 images)
ITERATIONS     = 1000                            # Full attack optimization steps
# ─────────────────────────────────────────────────────────────────────────


class DummyNoDefense:
    """Pass-through defense class for evaluating the baseline without defense."""
    def protect(self, x):
        return x


def denormalize(inputs, dataset_name, device):
    if dataset_name == 'CIFAR10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
        std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)
    else:
        mean = torch.tensor([0.1307]).view(1, 1, 1, 1).to(device)
        std  = torch.tensor([0.3081]).view(1, 1, 1, 1).to(device)
    return torch.clamp(inputs * std + mean, 0, 1)


def run_attack_against_defense(client_model, defense, test_loader, device,
                                dataset_name, max_images, iterations):
    attacker = WhiteBoxInversionAttack(
        client_model=client_model,
        dataset=dataset_name,
        iterations=iterations,
        lr=1e-2
    )
    tracker = AttackMetricsTracker()
    images_processed = 0

    for inputs, _ in test_loader:
        if images_processed >= max_images:
            break
        inputs = inputs.to(device)
        remaining = max_images - images_processed
        inputs = inputs[:remaining]

        with torch.no_grad():
            true_smashed = client_model(inputs)
            defended_smashed = defense.protect(true_smashed)

        reconstructed_batch = attacker.reconstruct(defended_smashed, inputs.shape)
        inputs_denorm = denormalize(inputs, dataset_name, device)

        tracker.log_batch(inputs_denorm, reconstructed_batch)
        images_processed += inputs.shape[0]

    return tracker.get_summary()


if __name__ == "__main__":
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()

    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    client_model = ClientModel(in_channels=in_channels).to(device)
    server_model = ServerModel(num_classes=Config.NUM_CLASSES).to(device)

    checkpoint_path = f"{Config.SAVE_DIR}/best_vanilla_sl_{Config.DATASET}.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            f"Run 'python main.py' first to train and save a model."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    client_model.load_state_dict(checkpoint['client_state'])
    server_model.load_state_dict(checkpoint['server_state'])
    print(f"[✓] Loaded checkpoint: {checkpoint_path}\n")

    trainer = SplitLearningTrainer(
        client_model=client_model,
        server_model=server_model,
        train_loader=train_loader,
        test_loader=test_loader
    )

    # ── 1. BASELINE EVALUATION (NO DEFENSE) ──────────────────────────────
    print("="*60)
    print("  BASELINE (NO DEFENSE)")
    print("="*60)
    no_defense = DummyNoDefense()
    baseline_acc = trainer.evaluate_with_defense(no_defense.protect)
    print(f"  Accuracy (no defense): {baseline_acc:.2f}%")

    print(f"  Running Baseline White-Box Inversion Attack...")
    baseline_summary = run_attack_against_defense(
        client_model, no_defense, test_loader, device,
        Config.DATASET, MAX_IMAGES, ITERATIONS
    )
    print(f"  Baseline PSNR: {baseline_summary['mean_psnr']:.2f} dB")
    print(f"  Baseline SSIM: {baseline_summary['mean_ssim']:.4f}\n")

    results = [{
        "epsilon": "inf (No Defense)",
        "accuracy": baseline_acc,
        "psnr": baseline_summary['mean_psnr'],
        "ssim": baseline_summary['mean_ssim']
    }]

    # ── 2. DEFENSE PARAMETER SWEEP ─────────────────────────────────────────
    for eps in EPSILON_VALUES:
        print("="*60)
        print(f"  DP-SL DEFENSE — epsilon = {eps}")
        print("="*60)

        defense = DPSLDefense(epsilon=eps, delta=DELTA, clip_norm=CLIP_NORM)
        print(f"  {defense}")

        acc = trainer.evaluate_with_defense(defense.protect)
        print(f"  Accuracy WITH defense: {acc:.2f}%  "
              f"(drop of {baseline_acc - acc:.2f} points from baseline)")

        print(f"  Running White-Box Inversion Attack against defended data...")
        summary = run_attack_against_defense(
            client_model, defense, test_loader, device,
            Config.DATASET, MAX_IMAGES, ITERATIONS
        )
        print(f"  PSNR (privacy, lower=better): {summary['mean_psnr']:.2f} dB")
        print(f"  SSIM (privacy, lower=better): {summary['mean_ssim']:.4f}\n")

        results.append({
            "epsilon": eps,
            "accuracy": acc,
            "psnr": summary['mean_psnr'],
            "ssim": summary['mean_ssim']
        })

    # ── 3. SAVE TO CSV ──────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    output_path = f"{Config.RESULTS_DIR}/dpsl_defense_evaluation_{Config.DATASET}.csv"
    df.to_csv(output_path, index=False)

    # ── 4. CUSTOM FORMATTED PRINT ──────────────────────────────────────────
    base_acc  = results[0]["accuracy"]
    base_psnr = results[0]["psnr"]
    base_ssim = results[0]["ssim"]

    # Pick eps = 200 for Executive Summary
    opt_row = next((r for r in results if r["epsilon"] == 200), results[1])
    eps_val = opt_row["epsilon"]
    dpsl_label = f"DP-SL Defense (eps = {float(eps_val):.1f})"

    print("\n" + "="*70)
    print(" 1. METHOD COMPARISON")
    print("="*70)
    print(f"{'Method / Setting':<28} {'Accuracy (%)':<14} {'PSNR (dB)':<12} {'SSIM':<8}")
    print("-" * 65)
    print(f"{'Vanilla SL (No Defense)':<28} {base_acc:<14.2f} {base_psnr:<12.2f} {base_ssim:<8.4f}")
    print(f"{dpsl_label:<28} {opt_row['accuracy']:<14.2f} {opt_row['psnr']:<12.2f} {opt_row['ssim']:<8.4f}")
    print("-" * 65)
    print(f"{'Defense Impact (Delta)':<28} ↓ {base_acc - opt_row['accuracy']:<12.2f}% ↓ {base_psnr - opt_row['psnr']:<10.2f}dB ↓ {base_ssim - opt_row['ssim']:<8.4f}\n")

    print("="*85)
    print(" 2. THE COMPLETE THESIS TABLE (ATTACK VS DP-SL DEFENSE)")
    print("="*85)
    print(f"{'Epsilon (ε)':<18} {'Accuracy (%)':<15} {'Δ Acc (%)':<12} {'PSNR (dB)':<12} {'Δ PSNR':<10} {'SSIM':<10} {'Δ SSIM':<8}")
    print("-" * 85)
    for r in results:
        eps_str = str(r["epsilon"])
        if "inf" in eps_str:
            print(f"{'inf (No Defense)':<18} {r['accuracy']:<15.2f} {'—':<12} {r['psnr']:<12.2f} {'—':<10} {r['ssim']:<10.4f} {'—':<8}")
        else:
            d_acc  = r['accuracy'] - base_acc
            d_psnr = r['psnr'] - base_psnr
            d_ssim = r['ssim'] - base_ssim
            print(f"{float(r['epsilon']):<18.1f} {r['accuracy']:<15.2f} {d_acc:<12.2f} {r['psnr']:<12.2f} {d_psnr:<10.2f} {r['ssim']:<10.4f} {d_ssim:<8.4f}")
    print("="*85)
    print(f"\nSaved raw data → {output_path}")