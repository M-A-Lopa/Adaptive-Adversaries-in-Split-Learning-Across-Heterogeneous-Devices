import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
from config import Config
from dataset import DatasetLoader
from all_split_learning.split_learning import SplitLearningTrainer
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel


def plot_results(train_losses, train_accuracies, test_accuracies, dataset_name):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    ax1.set_title('Training Loss', fontsize=13)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accuracies, 'b-', linewidth=2, label='Train Accuracy')
    ax2.plot(epochs, test_accuracies,  'r-', linewidth=2, label='Test Accuracy')
    ax2.set_title('Model Accuracy', fontsize=13)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{Config.MODEL_NAME} Split Learning — {dataset_name}', fontsize=14)
    plt.tight_layout()

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    save_path = f"{Config.RESULTS_DIR}/training_curves_{Config.MODEL_NAME.lower()}_{dataset_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
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
        server_model = PyramidCNNServerModel(cut_layer=Config.CUT_LAYER, num_classes=Config.NUM_CLASSES, in_channels=in_channels).to(device)
    else:
        client_model = ClientModel(in_channels=in_channels).to(device)
        server_model = ServerModel(num_classes=Config.NUM_CLASSES).to(device)

    checkpoint_path = f"{Config.SAVE_DIR}/best_{Config.MODEL_NAME.lower()}_sl_{Config.DATASET}.pth"

    if os.path.exists(checkpoint_path):
        print(f"[i] Checkpoint exists at {checkpoint_path}, but retraining anyway to log curves.")
    else:
        print(f"[i] No checkpoint found. Training {Config.MODEL_NAME} from scratch on {Config.DATASET}...")

    trainer = SplitLearningTrainer(
        client_model=client_model,
        server_model=server_model,
        train_loader=train_loader,
        test_loader=test_loader
    )
    trainer.train()
    trainer.save_results()

    plot_results(trainer.train_losses, trainer.train_accuracies, trainer.test_accuracies, Config.DATASET)