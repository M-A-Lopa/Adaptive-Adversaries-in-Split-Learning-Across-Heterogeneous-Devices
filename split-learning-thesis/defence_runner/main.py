import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from dataset import DatasetLoader
from all_split_learning.split_learning import SplitLearningTrainer
from all_attacks.attack_unsplit import UnSplitAttack
from all_attacks.attacks_whitebox import WhiteBoxInversionAttack, AttackMetricsTracker
from all_attacks.ae_decoder_attack import run_ae_decoder_attack, save_ae_attack_visualization
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel


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

    plt.suptitle(f'{Config.MODEL_NAME} Split Learning — {dataset_name}', fontsize=14)
    plt.tight_layout()

    save_path = f"{Config.RESULTS_DIR}/training_curves_{dataset_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    print(f"Training curves saved → {save_path}")


if __name__ == "__main__":

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()

    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    
    if Config.MODEL_NAME == "KAGN":
        client_model = KAGNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels, degree=Config.DEGREE).to(device)
        server_model = KAGNServerModel(cut_layer=Config.CUT_LAYER, num_classes=Config.NUM_CLASSES, in_channels=in_channels, degree=Config.DEGREE).to(device)
    elif Config.MODEL_NAME == "PyramidCNN":
        client_model = PyramidCNNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels).to(device)
        server_model = PyramidCNNServerModel(cut_layer=Config.CUT_LAYER, num_classes=Config.NUM_CLASSES,in_channels=in_channels).to(device)
    else:
        client_model = ClientModel(in_channels=in_channels).to(device)
        server_model = ServerModel(num_classes=Config.NUM_CLASSES).to(device)
        
        
    def build_fresh_client(): #Used by Unsplit
        if Config.MODEL_NAME == "KAGN":
            return KAGNClientModel(cut_layer=Config.CUT_LAYER,
                                   in_channels=in_channels, degree=3)
        elif Config.MODEL_NAME == "PyramidCNN":
            return PyramidCNNClientModel(cut_layer=Config.CUT_LAYER,
                                         in_channels=in_channels)
        else:
            return ClientModel(in_channels=in_channels)

    checkpoint_path = f"{Config.SAVE_DIR}/best_{Config.MODEL_NAME.lower()}_sl_{Config.DATASET}.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"\n[✓] Found existing trained weights for {Config.MODEL_NAME} at: {checkpoint_path}")
        print("Skipping training phase. Loading weights into network structures...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        client_model.load_state_dict(checkpoint['client_state'])
        server_model.load_state_dict(checkpoint['server_state'])
    else:
        print(f"\n[!] No local checkpoint found at: {checkpoint_path}")
        print(f"Initiating {Config.MODEL_NAME} Split Learning training pipeline...")
        
        trainer = SplitLearningTrainer(client_model=client_model, server_model=server_model, train_loader=train_loader, test_loader=test_loader)
        trainer.train()
        trainer.save_results()

    if Config.RUN_ATTACK:

        MAX_IMAGES  = 32
        ITERATIONS  = 1000

        if Config.DATASET == 'CIFAR10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(device)
            std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(device)
        else:
            mean = torch.tensor([0.1307]).view(1,1,1,1).to(device)
            std  = torch.tensor([0.3081]).view(1,1,1,1).to(device)

        all_results = {}

        # Attack 1: White-Box rMSE
        print("\n" + "="*60)
        print(f"  WHITE-BOX ATTACK — {Config.MODEL_NAME}")
        print("="*60)

        attacker = WhiteBoxInversionAttack(client_model=client_model,
                                           dataset=Config.DATASET,
                                           iterations=ITERATIONS, lr=1e-2)
        tracker = AttackMetricsTracker()
        images_processed = 0

        for inputs, _ in test_loader:
            if images_processed >= MAX_IMAGES:
                break
            inputs    = inputs.to(device)
            remaining = MAX_IMAGES - images_processed
            inputs    = inputs[:remaining]

            print(f"\n  Inverting {inputs.shape[0]} image(s) "
                  f"[{images_processed + inputs.shape[0]}/{MAX_IMAGES}]...")

            with torch.no_grad():
                target_smashed = client_model(inputs)
            reconstructed = attacker.reconstruct(target_smashed, inputs.shape)

            inputs_denorm = torch.clamp(inputs * std + mean, 0, 1)
            tracker.log_batch(inputs_denorm, reconstructed)
            images_processed += inputs.shape[0]

        wb_summary = tracker.get_summary()
        all_results['WhiteBox'] = wb_summary
        print(f"\n  PSNR: {wb_summary['mean_psnr']:.2f} dB | "
              f"SSIM: {wb_summary['mean_ssim']:.4f}")

        pd.DataFrame([wb_summary]).to_csv(
            f"{Config.RESULTS_DIR}/attack_whitebox_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv",
            index=False
        )

        #Attack 2: UnSplit (coordinate gradient descent, data-oblivious)
        print("\n" + "="*60)
        print(f"  UNSPLIT ATTACK — {Config.MODEL_NAME}")
        print("="*60)

        unsplit_attacker = UnSplitAttack(
            client_model=client_model,
            in_channels=in_channels,
            clone_builder=build_fresh_client
        )
        unsplit_summary = unsplit_attacker.run_attack(
            test_loader, num_batches=MAX_IMAGES // 32 or 1,
            inversion_steps=ITERATIONS
        )
        all_results['UnSplit'] = unsplit_summary
        print(f"\n  PSNR: {unsplit_summary['psnr']:.2f} dB | "
              f"SSIM: {unsplit_summary['ssim']:.4f}")

        pd.DataFrame([unsplit_summary]).to_csv(
            f"{Config.RESULTS_DIR}/attack_unsplit_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv",
            index=False
        )

        #Attack 3: AE Decoder (model-based inversion)
        print("\n" + "="*60)
        print(f"  AE DECODER ATTACK — {Config.MODEL_NAME}")
        print("="*60)

        ae_summary, ae_orig, ae_recon = run_ae_decoder_attack(
            client_model=client_model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            dataset=Config.DATASET,
            preprocess_fn=None,
            ae_epochs=50,
            label=f'{Config.MODEL_NAME} AE Decoder'
        )
        all_results['AE_Decoder'] = ae_summary
        pd.DataFrame([ae_summary]).to_csv(
            f"{Config.RESULTS_DIR}/attack_ae_decoder_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv",
            index=False
        )

        if len(ae_orig) > 0:
            save_ae_attack_visualization(
                originals=ae_orig, reconstructed=ae_recon,
                baseline_summary=wb_summary,     
                defense_summary=ae_summary,
                defense_name=Config.MODEL_NAME,
                dataset=Config.DATASET,
                results_dir=Config.RESULTS_DIR
            )

        # ── Combined summary table ────────────────────────────────────────────
        print("\n" + "="*68)
        print(f"   ALL ATTACKS SUMMARY — {Config.MODEL_NAME} on {Config.DATASET}")
        print("="*68)
        print(f"{'Attack':<20} {'PSNR (dB)':>12} {'SSIM':>10}")
        print("-"*45)
        print(f"  {'White-Box':<18} {wb_summary['mean_psnr']:>12.2f} "
              f"{wb_summary['mean_ssim']:>10.4f}")
        print(f"  {'UnSplit':<18} {unsplit_summary['psnr']:>12.2f} "
              f"{unsplit_summary['ssim']:>10.4f}")
        print(f"  {'AE Decoder':<18} {ae_summary['mean_psnr']:>12.2f} "
              f"{ae_summary['mean_ssim']:>10.4f}")
        print("="*68)

        combined_path = f"{Config.RESULTS_DIR}/all_attacks_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv"
        pd.DataFrame({
            'attack': ['WhiteBox', 'UnSplit', 'AE_Decoder'],
            'psnr': [wb_summary['mean_psnr'], unsplit_summary['psnr'], ae_summary['mean_psnr']],
            'ssim': [wb_summary['mean_ssim'], unsplit_summary['ssim'], ae_summary['mean_ssim']]
        }).to_csv(combined_path, index=False)
        print(f"\n  Combined results saved → {combined_path}")