import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from config import Config
from dataset import DatasetLoader
from all_model.models import ClientModel
from all_model.kagn_models import KAGNClientModel
from all_model.pyramid_cnn import PyramidCNNClientModel
from all_attacks.fsha_attack import FSHAAttack

HIJACK_EPOCHS   = 5
NUM_EVAL_IMAGES = 32
CRITIC_ITERS    = 5

MODELS_TO_RUN = ["Vanilla", "PyramidCNN", "KAGN"]

def build_client(model_name, in_channels):
    if model_name == "KAGN":
        return KAGNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels, degree=Config.DEGREE)
    elif model_name == "PyramidCNN":
        return PyramidCNNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels)
    else:
        return ClientModel(in_channels=in_channels)

if __name__ == "__main__":
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    private_loader, public_loader = dataset.get_loaders()

    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    results = []

    for model_name in MODELS_TO_RUN:
        print("\n" + "#" * 70)
        print(f"#   FSHA ATTACK -- MODEL: {model_name}  |  DATASET: {Config.DATASET}")
        print("#" * 70)

        client_model = build_client(model_name, in_channels).to(device)
        pilot_builder = lambda: build_client(model_name, in_channels)

        fsha = FSHAAttack(
            client_model=client_model,
            in_channels=in_channels,
            dataset=Config.DATASET,
            pilot_builder=pilot_builder,
            critic_iters=CRITIC_ITERS,
        )
        fsha.hijack(private_loader, public_loader, epochs=HIJACK_EPOCHS)
        summary = fsha.reconstruct(private_loader, num_images=NUM_EVAL_IMAGES)

        os.rename(
            f"{Config.RESULTS_DIR}/fsha_no_defense.png",
            f"{Config.RESULTS_DIR}/fsha_{model_name.lower()}_{Config.DATASET}.png"
        )

        results.append({
            "model": model_name,
            "psnr": summary['psnr'],
            "ssim": summary['ssim'],
            "mse":  summary['mse'],
        })

    df = pd.DataFrame(results)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    output_path = f"{Config.RESULTS_DIR}/fsha_all_models_{Config.DATASET}.csv"
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 68)
    print(f"   FSHA (PHASE 1: NO DEFENCE) -- ALL MODELS -- {Config.DATASET}")
    print("=" * 68)
    print(f"{'Model':<15} {'PSNR (dB)':>12} {'SSIM':>10} {'MSE':>10}")
    print("-" * 50)
    for r in results:
        print(f"  {r['model']:<13} {r['psnr']:>12.2f} {r['ssim']:>10.4f} {r['mse']:>10.5f}")
    print("=" * 68)
    print(f"\nSaved raw data -> {output_path}")