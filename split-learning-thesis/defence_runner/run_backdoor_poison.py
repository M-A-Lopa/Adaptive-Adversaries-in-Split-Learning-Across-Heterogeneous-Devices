import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset

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
SURROGATE_EPOCHS = 5
AUX_FRACTION     = 0.1   # share of the TRAIN set the malicious server uses as auxiliary data

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


def make_aux_loader(train_loader):
    """Auxiliary data for the malicious server: a fixed random slice of the training set.
    (The test set must not be used here, because ASR/CDA are measured on it.)"""
    base = train_loader.dataset
    g = torch.Generator().manual_seed(0)
    n = int(AUX_FRACTION * len(base))
    idx = torch.randperm(len(base), generator=g)[:n].tolist()
    return DataLoader(Subset(base, idx), batch_size=train_loader.batch_size, shuffle=True)


def run_one(model_name, mode, train_loader, test_loader, aux_loader, base_dataset, in_channels):
    tag = CLEAN_CHECKPOINT_TAGS.get(model_name, model_name.lower())
    cut_str = "fixed" if model_name == "Vanilla" else str(Config.CUT_LAYER)

    print("\n" + "#" * 70)
    print(f"#   BACKDOOR POISONING ({mode.upper()}) -- MODEL: {model_name}  |  "
          f"CUT: {cut_str}  |  DATASET: {Config.DATASET}")
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
        model_tag=tag,
    )
    print(f"  Poisoned checkpoint path: {attack._checkpoint_path()}")

    if attack.load_checkpoint():
        print("    Skipping training, evaluating the saved poisoned model...")
    else:
        clean_ckpt_path = f"{Config.SAVE_DIR}/best_{tag}_{Config.DATASET}.pth"
        attack.load_clean_init(clean_ckpt_path)

        if mode == "server":
            attack.pretrain_server_backdoor(aux_loader, epochs=SURROGATE_EPOCHS)

        attack.train(train_loader, test_loader, epochs=TRAIN_EPOCHS)
        attack.save_visualization(tag="no_defense")

        # continue with the BEST epoch (the one ProtoGuard will load), not the last one
        attack.load_checkpoint()

    cda, asr = attack.evaluate(test_loader)
    print(f"\n  Final (best checkpoint) -- ASR: {asr:.2f}%  CDA: {cda:.2f}%")

    cut_tag = "" if model_name == "Vanilla" else f"_cut{Config.CUT_LAYER}"
    pd.DataFrame(attack.history).to_csv(
        f"{Config.RESULTS_DIR}/backdoor_poison_epochs_{mode}_{model_name.lower()}"
        f"{cut_tag}_{Config.DATASET}.csv",
        index=False,
    )

    return {"model": model_name, "cut_layer": cut_str, "mode": mode, "asr": asr, "cda": cda}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut", nargs="+", type=int, default=[Config.CUT_LAYER],
                        help="cut layer(s), e.g. --cut 2 3")
    parser.add_argument("--model", nargs="+", default=MODELS_TO_RUN,
                        choices=["Vanilla", "PyramidCNN", "KAGN"])
    parser.add_argument("--mode", nargs="+", default=MODES_TO_RUN, choices=["client", "server"])
    parser.add_argument("--dataset", default=Config.DATASET, choices=["MNIST", "CIFAR10"])
    args = parser.parse_args()
    Config.DATASET = args.dataset

    print("=" * 60)
    print("  BACKDOOR POISONING EXPERIMENT")
    print(f"  Dataset : {Config.DATASET}")
    print(f"  Models  : {args.model}")
    print(f"  Cuts    : {args.cut}")
    print(f"  Modes   : {args.mode}")
    print("=" * 60)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"  Device  : {device}\n")

    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()
    base_dataset = train_loader.dataset
    aux_loader = make_aux_loader(train_loader)

    in_channels = 1 if Config.DATASET == 'MNIST' else 3

    results = []
    vanilla_done = False
    for cut in args.cut:
        Config.CUT_LAYER = cut
        for model_name in args.model:
            if model_name == "Vanilla":
                if vanilla_done:        # Vanilla has a fixed split -- run it only once
                    continue
                vanilla_done = True
            for mode in args.mode:
                results.append(run_one(model_name, mode, train_loader, test_loader,
                                       aux_loader, base_dataset, in_channels))

    df = pd.DataFrame(results)
    output_path = f"{Config.RESULTS_DIR}/backdoor_poison_all_models_{Config.DATASET}.csv"
    if os.path.exists(output_path):     # keep earlier runs, replace same model/cut/mode
        old = pd.read_csv(output_path, dtype={"cut_layer": str})
        if "cut_layer" in old.columns:
            df = pd.concat([old, df]).drop_duplicates(
                subset=["model", "cut_layer", "mode"], keep="last")
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"   BACKDOOR POISONING (NO DEFENSE) -- {Config.DATASET}")
    print("=" * 60)
    print(f"{'Model':<12} {'Cut':>5} {'Mode':<8} {'ASR (%)':>10} {'CDA (%)':>10}")
    print("-" * 60)
    for r in df.to_dict("records"):
        print(f"  {r['model']:<10} {str(r['cut_layer']):>5} {r['mode']:<8} "
              f"{r['asr']:>10.2f} {r['cda']:>10.2f}")
    print("=" * 60)
    print(f"\nSaved raw data -> {output_path}")
    print("\n  Backdoor poisoning experiment complete.")