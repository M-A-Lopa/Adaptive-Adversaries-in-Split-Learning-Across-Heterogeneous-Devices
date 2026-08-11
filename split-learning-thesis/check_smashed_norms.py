import os
import torch
from config import Config
from dataset import DatasetLoader
from models import ClientModel

device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')

dataset = DatasetLoader(dataset_name=Config.DATASET)
train_loader, test_loader = dataset.get_loaders()

in_channels = 1 if Config.DATASET == 'MNIST' else 3
client_model = ClientModel(in_channels=in_channels).to(device)

checkpoint_path = f"{Config.SAVE_DIR}/best_vanilla_sl_{Config.DATASET}.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
client_model.load_state_dict(checkpoint['client_state'])
client_model.eval()

all_norms = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        smashed = client_model(inputs)
        flat = smashed.view(smashed.shape[0], -1)
        norms = flat.norm(p=2, dim=1)
        all_norms.extend(norms.cpu().tolist())

all_norms = torch.tensor(all_norms)
print(f"Dataset: {Config.DATASET}")
print(f"Smashed data shape per sample: {tuple(smashed.shape[1:])}")
print(f"Mean L2 norm   : {all_norms.mean().item():.4f}")
print(f"Median L2 norm : {all_norms.median().item():.4f}")
print(f"Min L2 norm    : {all_norms.min().item():.4f}")
print(f"Max L2 norm    : {all_norms.max().item():.4f}")
print(f"90th percentile: {torch.quantile(all_norms, 0.9).item():.4f}")