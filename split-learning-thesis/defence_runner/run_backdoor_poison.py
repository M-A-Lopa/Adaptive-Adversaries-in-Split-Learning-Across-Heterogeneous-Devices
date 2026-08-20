import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from config import Config
from dataset import DatasetLoader
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel
from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack

TARGET_LABEL     = 0
POISON_RATE      = 0.05
PATCH_SIZE       = 4
TRIGGER_VALUE    = 1.0
TRAIN_EPOCHS     = 10
SURROGATE_EPOCHS = 5   # server mode only

MODELS_TO_RUN = ["Vanilla", "PyramidCNN", "KAGN"]
MODES_TO_RUN  = ["client", "server"]


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

    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    results = []

    for model_name in MODELS_TO_RUN:
        for mode in MODES_TO_RUN:
            print("\n" + "#" * 70)
            print(f"#   BACKDOOR POISONING ({mode.upper()}) -- MODEL: {model_name}  |  DATASET: {Config.DATASET}")
            print("#" * 70)

            client_model, server_model = build_split_models(model_name, in_channels)

            surrogate_builder = None
            if mode == "server":
                # attacker-controlled clone of the client architecture; captured
                # arguments are evaluated immediately below (same iteration).
                surrogate_builder = (lambda mn=model_name, ic=in_channels:
                                      build_split_models(mn, ic)[0])

            attack = BackdoorPoisonAttack(
                client_model=client_model,
                server_model=server_model,
                base_dataset=base_dataset,
                dataset=Config.DATASET,
                num_classes=Config.NUM_CLASSES,
                mode=mode,
                target_label=TARGET_LABEL,
                poison_rate=POISON_RATE,
                patch_size=PATCH_SIZE,
                trigger_value=TRIGGER_VALUE,
                surrogate_builder=surrogate_builder,
            )

            if mode == "server":
                attack.pretrain_server_backdoor(test_loader, epochs=SURROGATE_EPOCHS)

            attack.train(train_loader, test_loader, epochs=TRAIN_EPOCHS)

            summary = attack.summarise()
            attack.save_visualization(tag=f"{model_name.lower()}_{Config.DATASET}")

            pd.DataFrame(attack.history).to_csv(
                f"{Config.RESULTS_DIR}/backdoor_poison_epochs_{mode}_{model_name.lower()}_{Config.DATASET}.csv",
                index=False
            )

            results.append({
                "model": model_name,
                "mode": mode,
                "asr": summary['asr'],
                "cda": summary['cda'],
            })

    df = pd.DataFrame(results)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    output_path = f"{Config.RESULTS_DIR}/backdoor_poison_all_models_{Config.DATASET}.csv"
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 72)
    print(f"   BACKDOOR POISONING (CLIENT vs SERVER) -- ALL MODELS -- {Config.DATASET}")
    print("=" * 72)
    print(f"{'Model':<15} {'Mode':<10} {'ASR (%)':>10} {'CDA (%)':>10}")
    print("-" * 50)
    for r in results:
        print(f"  {r['model']:<13} {r['mode']:<10} {r['asr']:>10.2f} {r['cda']:>10.2f}")
    print("=" * 72)
    print(f"\nSaved raw data -> {output_path}")