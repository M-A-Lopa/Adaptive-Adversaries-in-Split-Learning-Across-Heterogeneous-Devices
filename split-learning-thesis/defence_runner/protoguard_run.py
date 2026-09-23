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
from all_defences.protoguard_sl_defense import ProtoGuardSLDefense

TARGET_LABEL  = 0
POISON_RATE   = 0.05
PATCH_SIZE    = 4
TRIGGER_VALUE = 1.0
ALPHA         = 0.5  

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


def run_one(model_name, mode, device, train_loader, test_loader, base_dataset, in_channels):
    print("\n" + "#" * 70)
    print(f"#   PROTOGUARD-SL vs BACKDOOR POISONING ({mode.upper()}) "
          f"-- MODEL: {model_name}  |  DATASET: {Config.DATASET}")
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
        model_tag=CLEAN_CHECKPOINT_TAGS.get(model_name, model_name.lower()),
    )

    if not attack.load_checkpoint():
        print(f"\n[!] No poisoned checkpoint found at: {attack._checkpoint_path()}")
        print("    Run defence_runner/run_backdoor_poison.py first to produce it. Skipping.")
        return None

    no_defense = DummyNoDefense()
    baseline_cda, baseline_asr = attack.evaluate(test_loader, defense=no_defense)
    print(f"\n  No Defense      -- CDA: {baseline_cda:.2f}%  ASR: {baseline_asr:.2f}%")

    defense = ProtoGuardSLDefense(num_classes=Config.NUM_CLASSES, alpha=ALPHA)
    print("\n  Fitting ProtoGuard-SL on labeled training data (class prototypes)...")
    defense.fit(attack.client_model, train_loader, device)

    defended_cda, defended_asr = attack.evaluate(test_loader, defense=defense)
    print(f"  ProtoGuard-SL   -- CDA: {defended_cda:.2f}%  ASR: {defended_asr:.2f}%  "
          f"(ASR drop: {baseline_asr - defended_asr:.2f} pts)")

    return {
        "model": model_name,
        "mode": mode,
        "no_defense_cda": baseline_cda,
        "no_defense_asr": baseline_asr,
        "protoguard_cda": defended_cda,
        "protoguard_asr": defended_asr,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  PROTOGUARD-SL DEFENSE EXPERIMENT")
    print(f"  Dataset : {Config.DATASET}")
    print(f"  alpha   : {ALPHA} (paper default)")
    print("=" * 60)
    print("  Note: no official ProtoGuard-SL repo exists (arXiv:2604.03595,")
    print("  Apr. 2026) -- this is an independent re-implementation of the")
    print("  paper's Algorithm 1.")
    print("=" * 60)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"  Device  : {device}\n")

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()
    base_dataset = train_loader.dataset

    in_channels = 1 if Config.DATASET == 'MNIST' else 3

    results = []
    for model_name in MODELS_TO_RUN:
        for mode in MODES_TO_RUN:
            summary = run_one(model_name, mode, device, train_loader, test_loader,
                              base_dataset, in_channels)
            if summary is not None:
                results.append(summary)

    if not results:
        print("\n  No poisoned checkpoints were found -- nothing to evaluate.")
        print("  Run defence_runner/run_backdoor_poison.py first.")
    else:
        df = pd.DataFrame(results)
        output_path = f"{Config.RESULTS_DIR}/protoguard_sl_defense_evaluation_{Config.DATASET}.csv"
        df.to_csv(output_path, index=False)

        print("\n" + "=" * 95)
        print(f"   PROTOGUARD-SL DEFENSE EVALUATION -- ALL MODELS -- {Config.DATASET}")
        print("=" * 95)
        print(f"{'Model':<12} {'Mode':<8} {'No-Def CDA':>11} {'No-Def ASR':>11} "
              f"{'ProtoGuard CDA':>15} {'ProtoGuard ASR':>15}")
        print("-" * 95)
        for r in results:
            print(f"  {r['model']:<10} {r['mode']:<8} "
                  f"{r['no_defense_cda']:>11.2f} {r['no_defense_asr']:>11.2f} "
                  f"{r['protoguard_cda']:>15.2f} {r['protoguard_asr']:>15.2f}")
        print("=" * 95)
        print(f"\nSaved raw data -> {output_path}")
        print("\n  ProtoGuard-SL experiment complete.")
        print(f"  All outputs in: {Config.RESULTS_DIR}/")