import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from config import Config
from dataset import DatasetLoader
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel
from all_attacks.villain_backdoor_attack import VILLAINBackdoorAttack, build_indexed_loader

WARMUP_EPOCHS    = 5
INFERENCE_EPOCHS = 5
INJECTION_EPOCHS = 10
BATCH_SIZE       = 128
TARGET_LABEL     = 0
TRIGGER_BETA     = 1.0
TRIGGER_FRACTION = 0.5
DROPOUT_KEEP     = 0.75
GAMMA_LOW        = 0.6
GAMMA_HIGH       = 1.2
THETA            = None
POISON_RATE      = 0.01
CANDIDATES       = 14

MODELS_TO_RUN = ["Vanilla", "PyramidCNN", "KAGN"]


def build_split_models(model_name, in_channels):
    if model_name == "KAGN":
        client = KAGNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels, degree=Config.DEGREE)
        server = KAGNServerModel(cut_layer=Config.CUT_LAYER, num_classes=Config.NUM_CLASSES,
                                 in_channels=in_channels, degree=Config.DEGREE)
    elif model_name == "PyramidCNN":
        client = PyramidCNNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels)
        server = PyramidCNNServerModel(cut_layer=Config.CUT_LAYER, num_classes=Config.NUM_CLASSES,
                                       in_channels=in_channels)
    else:
        client = ClientModel(in_channels=in_channels)
        server = ServerModel(num_classes=Config.NUM_CLASSES)
    return client, server


if __name__ == "__main__":
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()
    base_dataset = train_loader.dataset

    indexed_loader = build_indexed_loader(base_dataset, batch_size=BATCH_SIZE, shuffle=True)

    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    results = []

    for model_name in MODELS_TO_RUN:
        print("\n" + "#" * 70)
        print(f"#   VILLAIN BACKDOOR ATTACK -- MODEL: {model_name}  |  DATASET: {Config.DATASET}")
        print("#" * 70)

        client_model, server_model = build_split_models(model_name, in_channels)

        attack = VILLAINBackdoorAttack(
            client_model=client_model,
            server_model=server_model,
            base_dataset=base_dataset,
            dataset=Config.DATASET,
            num_classes=Config.NUM_CLASSES,
            target_label=TARGET_LABEL,
            beta=TRIGGER_BETA,
            trigger_fraction=TRIGGER_FRACTION,
            dropout_keep=DROPOUT_KEEP,
            gamma_low=GAMMA_LOW,
            gamma_high=GAMMA_HIGH,
            theta=THETA,
            poison_rate=POISON_RATE,
            candidates_per_batch=CANDIDATES,
        )

        attack.warmup(indexed_loader, epochs=WARMUP_EPOCHS)
        clean_baseline, _ = attack.evaluate(test_loader)
        print(f"  Clean data accuracy before attack: {clean_baseline:.2f}%")

        inference_accuracy = attack.infer_labels(indexed_loader, epochs=INFERENCE_EPOCHS)
        print(f"  Label inference accuracy: {100.0 * inference_accuracy:.2f}%")

        attack.fabricate_trigger(indexed_loader)
        attack.inject_backdoor(indexed_loader, test_loader, epochs=INJECTION_EPOCHS)

        summary = attack.summarise(test_loader=test_loader, clean_baseline=clean_baseline)
        attack.save_visualization(tag=f"{model_name.lower()}_{Config.DATASET}")

        pd.DataFrame(attack.history).to_csv(
            f"{Config.RESULTS_DIR}/villain_epochs_{model_name.lower()}_{Config.DATASET}.csv",
            index=False
        )

        results.append({
            "model": model_name,
            "lia": summary['lia'],
            "asr": summary['asr'],
            "cda": summary['cda'],
            "clean_baseline": clean_baseline,
            "inferred_samples": summary['inferred_samples'],
        })

    df = pd.DataFrame(results)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    output_path = f"{Config.RESULTS_DIR}/villain_all_models_{Config.DATASET}.csv"
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 72)
    print(f"   VILLAIN (PHASE 1: NO DEFENCE) -- ALL MODELS -- {Config.DATASET}")
    print("=" * 72)
    print(f"{'Model':<15} {'LIA (%)':>10} {'ASR (%)':>10} {'CDA (%)':>10} {'Clean (%)':>11}")
    print("-" * 60)
    for r in results:
        print(f"  {r['model']:<13} {r['lia']:>10.2f} {r['asr']:>10.2f} {r['cda']:>10.2f} "
              f"{r['clean_baseline']:>11.2f}")
    print("=" * 72)
    print(f"\nSaved raw data -> {output_path}")