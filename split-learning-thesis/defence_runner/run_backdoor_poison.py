import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd

from config import Config
from dataset import DatasetLoader
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel
from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack, DummyNoDefense

TARGET_LABEL     = 0
POISON_RATE      = 0.05
PATCH_SIZE       = 4
TRIGGER_VALUE    = 1.0
TRAIN_EPOCHS     = 10
SURROGATE_EPOCHS = 5  

MODELS_TO_RUN = ["Vanilla", "PyramidCNN", "KAGN"]
MODES_TO_RUN  = ["client", "server"]

CLEAN_CHECKPOINT_TAGS = {
    "Vanilla":    "vanilla_sl",
    "PyramidCNN": "pyramidcnn_sl",
    "KAGN":       "kagn_sl",
}


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


def run_one(model_name, mode, device, train_loader, test_loader, base_dataset, in_channels, defense):
    print("\n" + "#" * 70)
    print(f"#   BACKDOOR POISONING ({mode.upper()}) -- MODEL: {model_name}  |  DATASET: {Config.DATASET}")
    print("#" * 70)

    client_model, server_model = build_split_models(model_name, in_channels)

    surrogate_builder = None
    if mode == "server":

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
        defense=defense,
        model_tag=CLEAN_CHECKPOINT_TAGS.get(model_name, model_name.lower()),
    )


    clean_ckpt_path = f"{Config.SAVE_DIR}/best_{CLEAN_CHECKPOINT_TAGS.get(model_name, model_name.lower())}_{Config.DATASET}.pth"
    attack.load_clean_init(clean_ckpt_path)

    if mode == "server":
        attack.pretrain_server_backdoor(test_loader, epochs=SURROGATE_EPOCHS)

    if attack.load_checkpoint():
        print("    Skipping training phase, evaluating loaded checkpoint...")
        clean_acc, asr = attack.evaluate(test_loader)
        attack.history['epoch'].append(0)
        attack.history['asr'].append(asr)
        attack.history['cda'].append(clean_acc)
    else:
        attack.train(train_loader, test_loader, epochs=TRAIN_EPOCHS)

    summary = attack.summarise()
    attack.save_visualization(tag=f"{model_name.lower()}_{Config.DATASET}")

    pd.DataFrame(attack.history).to_csv(
        f"{Config.RESULTS_DIR}/backdoor_poison_epochs_{mode}_{model_name.lower()}_{Config.DATASET}.csv",
        index=False
    )

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("  BACKDOOR POISONING EXPERIMENT")
    print(f"  Dataset : {Config.DATASET}")
    print("=" * 60)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"  Device  : {device}\n")

    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()
    base_dataset = train_loader.dataset

    in_channels = 1 if Config.DATASET == 'MNIST' else 3

    defense = DummyNoDefense()

    results = []
    for model_name in MODELS_TO_RUN:
        for mode in MODES_TO_RUN:
            summary = run_one(model_name, mode, device, train_loader, test_loader,
                              base_dataset, in_channels, defense)
            results.append({
                "model": model_name,
                "mode": mode,
                "asr": summary['asr'],
                "cda": summary['cda'],
                "defense": summary['defense'],
            })

    df = pd.DataFrame(results)
    output_path = f"{Config.RESULTS_DIR}/backdoor_poison_all_models_{Config.DATASET}.csv"
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 78)
    print(f"   BACKDOOR POISONING (CLIENT vs SERVER) -- ALL MODELS -- {Config.DATASET}")
    print(f"   Defense: {defense}")
    print("=" * 78)
    print(f"{'Model':<15} {'Mode':<10} {'ASR (%)':>10} {'CDA (%)':>10}")
    print("-" * 50)
    for r in results:
        print(f"  {r['model']:<13} {r['mode']:<10} {r['asr']:>10.2f} {r['cda']:>10.2f}")
    print("=" * 78)
    print(f"\nSaved raw data -> {output_path}")
    print("\n  Backdoor poisoning experiment complete.")
    print(f"  All outputs in: {Config.RESULTS_DIR}/")