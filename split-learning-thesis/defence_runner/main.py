
# Run this file to start Vanilla Split Learning training
# Command: python main.py

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from dataset import DatasetLoader
from all_split_learning.split_learning import SplitLearningTrainer
from all_attacks.attacks_whitebox import WhiteBoxInversionAttack, AttackMetricsTracker
from all_model.models import ClientModel, ServerModel
#from pgsl_defense import PGSLDefenseModules



def plot_results(train_losses, train_accuracies, test_accuracies, dataset_name):
    
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    ax1.set_title('Training Loss', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accuracies, 'b-', linewidth=2, label='Train Accuracy')
    ax2.plot(epochs, test_accuracies,  'r-', linewidth=2, label='Test Accuracy')
    ax2.set_title('Model Accuracy', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Vanilla Split Learning — {dataset_name}', fontsize=14)
    plt.tight_layout()

    save_path = f"{Config.RESULTS_DIR}/training_curves_{dataset_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    print(f"Training curves saved → {save_path}")


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # CONTROL FLAGS
    # -------------------------------------------------------------------------
    # Set to True when you want to execute the White-Box Inversion Attack
    RUN_ATTACK = True
    

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")

    # -------------------------------------------------------------------------
    # 1. LOAD DATA PIPELINES (Skips downloading if root folders exist)
    # -------------------------------------------------------------------------
    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()

    # -------------------------------------------------------------------------
    # 2. INSTANTIATE BASE MODEL ARCHITECTURE
    # -------------------------------------------------------------------------
    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    client_model = ClientModel(in_channels=in_channels).to(device)
    server_model = ServerModel(num_classes=Config.NUM_CLASSES).to(device)

    # -------------------------------------------------------------------------
    # 3. CHECK FOR LOCAL TRAINED WEIGHTS (Skips training loop if found)
    # -------------------------------------------------------------------------
    checkpoint_path = f"{Config.SAVE_DIR}/best_vanilla_sl_{Config.DATASET}.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"\n[✓] Found existing trained weights at: {checkpoint_path}")
        print("Skipping training phase. Loading weights into network structures...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        client_model.load_state_dict(checkpoint['client_state'])
        server_model.load_state_dict(checkpoint['server_state'])
    else:
        print(f"\n[!] No local checkpoint found at: {checkpoint_path}")
        print("Initiating Vanilla Split Learning engine training pipeline...")
        
        trainer = SplitLearningTrainer(
            client_model=client_model,
            server_model=server_model,
            train_loader=train_loader,
            test_loader=test_loader
        )
        # Train and save the checkpoint automatically
        trainer.train()
        trainer.save_results()

    # -------------------------------------------------------------------------
    # 4. CONDITIONAL MODEL INVERSION ATTACK RUN
    # -------------------------------------------------------------------------
    if RUN_ATTACK:
        print("\n" + "="*60)
        print("      EXECUTING MODEL INVERSION EVALUATION RUN")
        print("="*60)

        # ── Speed control — change these two numbers only ─────────────────
        MAX_IMAGES = 32    # Total images to reconstruct across all batches
                        # Keep at 8 for fast confirmation run
                        # Increase to 32-64 only for final thesis results
        ITERATIONS = 1000   # Gradient steps per image
                        # 300 = fast confirmation (2-3 min on RTX 3050)
                        # 1000 = thesis quality (10-15 min on RTX 3050)
        # ──────────────────────────────────────────────────────────────────

        # Initialize Attack Engine with controlled iteration count
        attacker = WhiteBoxInversionAttack(
            client_model=client_model,
            dataset=Config.DATASET,
            iterations=ITERATIONS,   # uses the variable above, not hardcoded
            lr=1e-2
        )
        tracker = AttackMetricsTracker()

        images_processed = 0  # Counter to stop after MAX_IMAGES

        for batch_idx, (inputs, _) in enumerate(test_loader):

            if images_processed >= MAX_IMAGES:
                break  # Stop once enough images are reconstructed

            inputs = inputs.to(device)

            # Only take as many images as needed from this batch
            remaining  = MAX_IMAGES - images_processed
            inputs     = inputs[:remaining]   # slice to exact number needed

            print(f"\nInverting {inputs.shape[0]} image(s) "
                f"[{images_processed + inputs.shape[0]}/{MAX_IMAGES} total]...")

            with torch.no_grad():
                target_smashed = client_model(inputs)

            reconstructed_batch = attacker.reconstruct(target_smashed, inputs.shape)

            # Denormalize inputs before metric comparison
            if Config.DATASET == 'CIFAR10':
                mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(device)
                std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(device)
            else:
                mean = torch.tensor([0.1307]).view(1,1,1,1).to(device)
                std  = torch.tensor([0.3081]).view(1,1,1,1).to(device)

            inputs_denorm = torch.clamp(inputs * std + mean, 0, 1)

            tracker.log_batch(inputs_denorm, reconstructed_batch)
            images_processed += inputs.shape[0]

        # Collate Summary Metrics and export results
        summary = tracker.get_summary()
        print("\n" + "="*60)
        print("      ATTACK RUN STRUCTURAL EVALUATION SUMMARY")
        print("="*60)
        print(f"Mean Peak Signal-to-Noise Ratio (PSNR) : {summary['mean_psnr']:.2f} dB")
        print(f"Mean Structural Similarity Index (SSIM): {summary['mean_ssim']:.4f}")
        print(f"Total images evaluated                 : {images_processed}")
        print("="*60)

        metrics_df = pd.DataFrame([summary])
        output_path = f"{Config.RESULTS_DIR}/attack_evaluation_{Config.DATASET}.csv"
        metrics_df.to_csv(output_path, index=False)
        print(f"Evaluation metrics logged successfully → {output_path}\n")