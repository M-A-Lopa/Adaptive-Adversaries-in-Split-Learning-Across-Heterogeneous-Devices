import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from config import Config
from dataset import DatasetLoader
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel
from all_attacks.label_leakage_attack import GradientNormLabelLeakageAttack, build_binary_split_loaders

TRAINING_EPOCHS = 5
POSITIVE_CLASS  = 0
POSITIVE_RATIO  = 0.1
BATCH_SIZE      = 128
LEARNING_RATE   = 1e-4

MODELS_TO_RUN = ["Vanilla", "PyramidCNN", "KAGN"]


def build_split_models(model_name, in_channels):
    if model_name == "KAGN":
        client = KAGNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels, degree=Config.DEGREE)
        server = KAGNServerModel(cut_layer=Config.CUT_LAYER, num_classes=1, in_channels=in_channels, degree=Config.DEGREE)
    elif model_name == "PyramidCNN":
        client = PyramidCNNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels)
        server = PyramidCNNServerModel(cut_layer=Config.CUT_LAYER, num_classes=1, in_channels=in_channels)
    else:
        client = ClientModel(in_channels=in_channels)
        server = ServerModel(num_classes=1)
    return client, server


if __name__ == "__main__":
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, _ = dataset.get_loaders()
    base_dataset = train_loader.dataset

    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    results = []

    for model_name in MODELS_TO_RUN:
        print("\n" + "#" * 70)
        print(f"#   LABEL LEAKAGE ATTACK -- MODEL: {model_name}  |  DATASET: {Config.DATASET}")
        print("#" * 70)

        binary_train_loader, binary_test_loader = build_binary_split_loaders(
            base_dataset,
            target_class=POSITIVE_CLASS,
            positive_ratio=POSITIVE_RATIO,
            batch_size=BATCH_SIZE,
        )

        client_model, server_model = build_split_models(model_name, in_channels)

        attack = GradientNormLabelLeakageAttack(
            client_model=client_model,
            server_model=server_model,
            dataset=Config.DATASET,
            target_class=POSITIVE_CLASS,
            learning_rate=LEARNING_RATE,
        )
        summary = attack.run(binary_train_loader, binary_test_loader, epochs=TRAINING_EPOCHS)
        attack.save_visualization(tag=f"{model_name.lower()}_{Config.DATASET}")

        pd.DataFrame(attack.history).to_csv(
            f"{Config.RESULTS_DIR}/label_leakage_batches_{model_name.lower()}_{Config.DATASET}.csv",
            index=False
        )

        results.append({
            "model": model_name,
            "norm_leak_auc_cut": summary['q95_norm_leak_auc_cut'],
            "cosine_leak_auc_cut": summary['q95_cosine_leak_auc_cut'],
            "norm_leak_auc_first": summary['q95_norm_leak_auc_first'],
            "cosine_leak_auc_first": summary['q95_cosine_leak_auc_first'],
            "majority_accuracy": summary['q95_majority_accuracy_cut'],
            "test_auc": summary['test_auc'],
        })

    df = pd.DataFrame(results)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    output_path = f"{Config.RESULTS_DIR}/label_leakage_all_models_{Config.DATASET}.csv"
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 78)
    print(f"   LABEL LEAKAGE (PHASE 1: NO DEFENCE) -- ALL MODELS -- {Config.DATASET}")
    print("=" * 78)
    print(f"{'Model':<15} {'Norm AUC':>11} {'Cosine AUC':>12} {'Majority Acc':>14} {'Test AUC':>10}")
    print("-" * 66)
    for r in results:
        print(f"  {r['model']:<13} {r['norm_leak_auc_cut']:>11.4f} {r['cosine_leak_auc_cut']:>12.4f} "
              f"{r['majority_accuracy']:>14.4f} {r['test_auc']:>10.4f}")
    print("=" * 78)
    print(f"\nSaved raw data -> {output_path}")