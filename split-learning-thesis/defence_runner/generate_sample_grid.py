# generate_sample_grid.py
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
from config import Config
from dataset import DatasetLoader

def denormalize(img, dataset_name):
    if dataset_name == 'CIFAR10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    else:
        mean = torch.tensor([0.1307]).view(1, 1, 1)
        std  = torch.tensor([0.3081]).view(1, 1, 1)
    return torch.clamp(img * std + mean, 0, 1)

def save_grid(dataset_name, n_per_class=4, out_path=None):
    loader = DatasetLoader(dataset_name=dataset_name)
    train_loader, _ = loader.get_loaders()
    classes = {}
    for imgs, labels in train_loader:
        for img, lab in zip(imgs, labels):
            lab = lab.item()
            classes.setdefault(lab, [])
            if len(classes[lab]) < n_per_class:
                classes[lab].append(img)
        if all(len(v) >= n_per_class for v in classes.values()) and len(classes) == 10:
            break

    fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class * 1.3, 10 * 1.3))
    for c in range(10):
        for j in range(n_per_class):
            img = denormalize(classes[c][j], dataset_name)
            img = img.permute(1, 2, 0).squeeze().numpy()
            ax = axes[c, j]
            ax.imshow(img, cmap='gray' if dataset_name == 'MNIST' else None)
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(str(c), rotation=0, labelpad=15, fontsize=10)
    plt.suptitle(f'{dataset_name} sample images', fontsize=13)
    plt.tight_layout()

    out_path = out_path or f"{Config.RESULTS_DIR}/sample_grid_{dataset_name}.png"
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    save_grid("CIFAR10")
    save_grid("MNIST")  