
# Run this file to start Vanilla Split Learning training
# Command: python main.py

import torch
import matplotlib.pyplot as plt
from config import Config
from dataset import DatasetLoader
from models import ClientModel, ServerModel
from split_learning import SplitLearningTrainer


def plot_results(train_losses, train_accuracies, test_accuracies):
    #Plots and saves training curves
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # -------------------Loss curve--------------------------
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    ax1.set_title('Training Loss', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -------------------Accuracy curve--------------------------
    ax2.plot(epochs, train_accuracies, 'b-', linewidth=2, label='Train Accuracy')
    ax2.plot(epochs, test_accuracies,  'r-', linewidth=2, label='Test Accuracy')
    ax2.set_title('Model Accuracy', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Vanilla Split Learning — {Config.DATASET}', fontsize=14)
    plt.tight_layout()

    save_path = f"{Config.RESULTS_DIR}/training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved → {save_path}")


if __name__ == "__main__":

    # --------------------Load dataset--------------------------
    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()

    # ----------------------Initialize models--------------------
    in_channels = 1 if Config.DATASET == 'MNIST' else 3

    client_model = ClientModel(in_channels=in_channels)
    server_model = ServerModel(num_classes=Config.NUM_CLASSES)

    print(f"\nClient parameters : {sum(p.numel() for p in client_model.parameters()):,}")
    print(f"Server parameters : {sum(p.numel() for p in server_model.parameters()):,}")

    # -----------------------Train-------------------------------
    trainer = SplitLearningTrainer(
        client_model=client_model,
        server_model=server_model,
        train_loader=train_loader,
        test_loader=test_loader
    )

    train_losses, train_accs, test_accs = trainer.train()

    # ------------------Save results and plot---------------------
    trainer.save_results()
    plot_results(train_losses, train_accs, test_accs)