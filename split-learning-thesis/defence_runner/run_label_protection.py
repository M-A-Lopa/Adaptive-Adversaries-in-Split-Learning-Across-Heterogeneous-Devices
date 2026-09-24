import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from config import Config
from dataset import DatasetLoader
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel
from all_attacks.label_leakage_attack import GradientNormLabelLeakageAttack, build_binary_split_loaders
from all_defences.label_protection_defense import LabelProtectionDefense, attach_label_protection

TRAINING_EPOCHS = 5
POSITIVE_CLASS  = 0
POSITIVE_RATIO  = 0.1
BATCH_SIZE      = 128
LEARNING_RATE   = 1e-4
SEED            = 0

RUN_PHASE_A_DEFENSE_SWEEP = True
RUN_PHASE_B_CUT_LAYER_SWEEP = True

MODELS_TO_RUN = ["Vanilla", "PyramidCNN", "KAGN"]

DEFENSES_TO_RUN = (
    [("no_noise", {})]
    + [("max_norm", {})]
    + [("iso", {"ratio": r}) for r in [4.5, 6.0, 9.0, 11.0, 13.0, 15.0]]
    + [("marvell", {"init_scale": s, "p_frac": "pos_frac", "uv_choice": "uv", "dynamic": False})
       for s in [0.05, 0.15, 0.25, 0.4, 1.5, 1.75]]
    + [("perp", {"lower": 1.0, "upper": 5.0})]
)

CUT_LAYER_SWEEP_MODELS = ["PyramidCNN", "KAGN"]
CUT_LAYERS_TO_SWEEP    = [1, 2, 3, 4, 5]
CUT_LAYER_SWEEP_DEFENSES = [
    ("no_noise", {}),
    ("max_norm", {}),
    ("iso", {"ratio": 9.0}),
    ("marvell", {"init_scale": 0.25, "p_frac": "pos_frac", "uv_choice": "uv", "dynamic": False}),
    ("marvell", {"init_scale": 1.75, "p_frac": "pos_frac", "uv_choice": "uv", "dynamic": False}),
]

REPRESENTATIVE_DEFENSE = "marvell_s1.75"

VULNERABILITY_LEVELS = [
    (0.95, "Critical Breach (near-exact label recovery)"),
    (0.80, "High Breach"),
    (0.65, "Moderate Breach"),
    (0.55, "Weak / Degraded"),
    (0.00, "Protected (near chance)"),
]


def two_means_predict(scores):
    scores = scores.detach().double().cpu()
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    n = sorted_scores.numel()
    predicted = torch.zeros(n, dtype=torch.float64)
    if n < 2 or sorted_scores[0] == sorted_scores[-1]:
        return predicted
    csum = torch.cumsum(sorted_scores, 0)
    csum_sq = torch.cumsum(sorted_scores ** 2, 0)
    k = torch.arange(1, n, dtype=torch.float64)
    left_sse = csum_sq[:-1] - csum[:-1] ** 2 / k
    right_sum = csum[-1] - csum[:-1]
    right_sse = (csum_sq[-1] - csum_sq[:-1]) - right_sum ** 2 / (n - k)
    split = int(torch.argmin(left_sse + right_sse).item()) + 1
    predicted[order[split:]] = 1.0
    return predicted


def recovery_accuracies(scores, labels):
    truth = labels.detach().cpu().double()
    predicted = two_means_predict(scores)
    accuracy = float((predicted == truth).double().mean().item())
    tpr = float(((predicted == 1) & (truth == 1)).double().sum() / (truth == 1).double().sum())
    tnr = float(((predicted == 0) & (truth == 0)).double().sum() / (truth == 0).double().sum())
    fpr_curve, tpr_curve, _ = roc_curve(truth.numpy(), scores.detach().cpu().double().numpy())
    return accuracy, 0.5 * (tpr + tnr), float(0.5 + 0.5 * np.max(tpr_curve - fpr_curve))


class GradientObserver:
    def __init__(self):
        self.history = {key: [] for key in [
            'cosine_leak_auc_cut_cleanref', 'chance_accuracy',
            'norm_recovery_acc', 'norm_recovery_bal_acc', 'norm_worstcase_bal_acc',
            'cosine_recovery_acc', 'cosine_recovery_bal_acc', 'cosine_worstcase_bal_acc']}

    def __call__(self, clean_gradient, sent_gradient, labels):
        clean = clean_gradient.flatten(1)
        sent = sent_gradient.flatten(1)
        positives = torch.nonzero(labels > 0.5).flatten()
        if positives.numel() == 0 or positives.numel() == labels.numel():
            return
        reference = clean[positives[torch.randint(len(positives), (1,))].item()]
        cosine = F.cosine_similarity(sent, reference.unsqueeze(0), dim=1)
        norm = sent.norm(p=2, dim=1)
        truth = labels.detach().cpu().numpy()

        self.history['cosine_leak_auc_cut_cleanref'].append(float(roc_auc_score(truth, cosine.cpu().numpy())))
        self.history['chance_accuracy'].append(float(1.0 - labels.mean().item()))
        for name, score in [('norm', norm), ('cosine', cosine)]:
            accuracy, balanced, worst_case = recovery_accuracies(score, labels)
            self.history[f'{name}_recovery_acc'].append(accuracy)
            self.history[f'{name}_recovery_bal_acc'].append(balanced)
            self.history[f'{name}_worstcase_bal_acc'].append(worst_case)

    def summary(self):
        out = {}
        for key, values in self.history.items():
            values = np.asarray(values, dtype=float)
            out[f'mean_{key}'] = float(values.mean()) if values.size else float('nan')
            out[f'q95_{key}'] = float(np.quantile(values, 0.95)) if values.size else float('nan')
        return out


class TrackedLabelLeakageAttack(GradientNormLabelLeakageAttack):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_history = {'epoch': [], 'test_accuracy': [], 'test_balanced_accuracy': [],
                              'test_auc': [], 'test_loss': []}

    def evaluate(self, test_loader):
        self.client_model.eval()
        self.server_model.eval()
        scores, truths = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                logits = self.server_model(self.client_model(inputs)).squeeze(1)
                scores.append(logits.cpu().numpy())
                truths.append(np.asarray(labels, dtype=np.float32))
        scores = np.concatenate(scores)
        truths = np.concatenate(truths)

        predicted = (scores > 0).astype(np.float32)
        positives, negatives = truths == 1, truths == 0
        tpr = float((predicted[positives] == 1).mean()) if positives.any() else float('nan')
        tnr = float((predicted[negatives] == 0).mean()) if negatives.any() else float('nan')
        auc = float('nan') if truths.min() == truths.max() else float(roc_auc_score(truths, scores))

        self.epoch_history['epoch'].append(len(self.epoch_history['epoch']) + 1)
        self.epoch_history['test_accuracy'].append(100.0 * float((predicted == truths).mean()))
        self.epoch_history['test_balanced_accuracy'].append(100.0 * 0.5 * (tpr + tnr))
        self.epoch_history['test_auc'].append(auc)
        self.epoch_history['test_loss'].append(float(F.binary_cross_entropy_with_logits(
            torch.from_numpy(scores), torch.from_numpy(truths)).item()))
        return auc


def build_split_models(model_name, in_channels, cut_layer):
    if model_name == "KAGN":
        client = KAGNClientModel(cut_layer=cut_layer, in_channels=in_channels, degree=Config.DEGREE)
        server = KAGNServerModel(cut_layer=cut_layer, num_classes=1, in_channels=in_channels, degree=Config.DEGREE)
    elif model_name == "PyramidCNN":
        client = PyramidCNNClientModel(cut_layer=cut_layer, in_channels=in_channels)
        server = PyramidCNNServerModel(cut_layer=cut_layer, num_classes=1, in_channels=in_channels)
    else:
        client = ClientModel(in_channels=in_channels)
        server = ServerModel(num_classes=1)
    return client, server


def run_single(model_name, cut_layer, method, params, in_channels, loaders, phase):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    defense = LabelProtectionDefense(method, **params)
    tag = defense.tag()
    cut_label = "fixed" if model_name == "Vanilla" else cut_layer

    print("\n" + "#" * 78)
    print(f"#   {phase} | MODEL: {model_name} | CUT LAYER: {cut_label} | DEFENSE: {tag} | {Config.DATASET}")
    print("#" * 78)

    original_cut_layer = Config.CUT_LAYER
    Config.CUT_LAYER = cut_layer
    try:
        client_model, server_model = build_split_models(model_name, in_channels, cut_layer)
        attack = TrackedLabelLeakageAttack(
            client_model=client_model,
            server_model=server_model,
            dataset=Config.DATASET,
            target_class=POSITIVE_CLASS,
            learning_rate=LEARNING_RATE,
        )
        observer = GradientObserver()
        protected = attach_label_protection(attack, defense, observer)
        summary = attack.run(loaders[0], loaders[1], epochs=TRAINING_EPOCHS)

        run_tag = f"{model_name.lower()}_cut{cut_label}_{tag}_{Config.DATASET}"
    finally:
        Config.CUT_LAYER = original_cut_layer

    pd.DataFrame(attack.history).to_csv(
        f"{Config.RESULTS_DIR}/label_protection_batches_{run_tag}.csv", index=False)
    pd.DataFrame(observer.history).to_csv(
        f"{Config.RESULTS_DIR}/label_protection_recovery_{run_tag}.csv", index=False)
    pd.DataFrame(attack.epoch_history).to_csv(
        f"{Config.RESULTS_DIR}/label_protection_epochs_{run_tag}.csv", index=False)
    if defense.method == "marvell":
        pd.DataFrame(defense.solver_log).to_csv(
            f"{Config.RESULTS_DIR}/label_protection_marvell_solver_{run_tag}.csv", index=False)

    extra = observer.summary()
    epochs = attack.epoch_history
    solver_ms = [entry['solver_ms'] for entry in defense.solver_log]

    return {
        "phase": phase,
        "model": model_name,
        "cut_layer": cut_label,
        "smashed_dim": int(np.prod(attack.smashed_shape[1:])),
        "method": defense.method,
        "defense": tag,
        "norm_leak_auc_cut": summary['q95_norm_leak_auc_cut'],
        "cosine_leak_auc_cut": summary['q95_cosine_leak_auc_cut'],
        "cosine_leak_auc_cut_cleanref": extra['q95_cosine_leak_auc_cut_cleanref'],
        "norm_leak_auc_first": summary['q95_norm_leak_auc_first'],
        "cosine_leak_auc_first": summary['q95_cosine_leak_auc_first'],
        "majority_counting_acc_pct": 100 * summary['mean_majority_accuracy_cut'],
        "majority_counting_acc_q95": summary['q95_majority_accuracy_cut'],
        "chance_accuracy_pct": 100 * extra['mean_chance_accuracy'],
        "norm_recovery_acc_pct": 100 * extra['mean_norm_recovery_acc'],
        "norm_recovery_bal_acc_pct": 100 * extra['mean_norm_recovery_bal_acc'],
        "norm_worstcase_bal_acc_pct": 100 * extra['mean_norm_worstcase_bal_acc'],
        "cosine_recovery_acc_pct": 100 * extra['mean_cosine_recovery_acc'],
        "cosine_recovery_bal_acc_pct": 100 * extra['mean_cosine_recovery_bal_acc'],
        "cosine_worstcase_bal_acc_pct": 100 * extra['mean_cosine_worstcase_bal_acc'],
        "test_accuracy_pct": epochs['test_accuracy'][-1],
        "best_test_accuracy_pct": float(np.nanmax(epochs['test_accuracy'])),
        "test_balanced_accuracy_pct": epochs['test_balanced_accuracy'][-1],
        "best_test_balanced_accuracy_pct": float(np.nanmax(epochs['test_balanced_accuracy'])),
        "test_auc": summary['test_auc'],
        "best_test_auc": float(np.nanmax(epochs['test_auc'])),
        "test_loss": epochs['test_loss'][-1],
        "defense_ms_per_batch": float(np.mean(protected.defense_ms)) if protected.defense_ms else float('nan'),
        "marvell_solver_ms_per_batch": float(np.mean(solver_ms)) if solver_ms else float('nan'),
    }


def vulnerability(*aucs):
    strongest = np.nanmax(aucs)
    for threshold, label in VULNERABILITY_LEVELS:
        if strongest >= threshold:
            return label
    return VULNERABILITY_LEVELS[-1][1]


DELTA_COLUMNS = [
    ("test_accuracy_pct", "delta_test_accuracy_pct"),
    ("best_test_accuracy_pct", "delta_best_test_accuracy_pct"),
    ("test_balanced_accuracy_pct", "delta_test_balanced_accuracy_pct"),
    ("test_auc", "delta_test_auc"),
    ("norm_leak_auc_cut", "delta_norm_leak_auc_cut"),
    ("cosine_leak_auc_cut", "delta_cosine_leak_auc_cut"),
    ("cosine_leak_auc_cut_cleanref", "delta_cosine_leak_auc_cut_cleanref"),
    ("norm_recovery_acc_pct", "delta_norm_recovery_acc_pct"),
    ("norm_worstcase_bal_acc_pct", "delta_norm_worstcase_bal_acc_pct"),
    ("cosine_worstcase_bal_acc_pct", "delta_cosine_worstcase_bal_acc_pct"),
]


def add_thesis_columns(rows):
    baselines = {(r['phase'], r['model'], r['cut_layer']): r for r in rows if r['method'] == 'no_noise'}
    for r in rows:
        base = baselines.get((r['phase'], r['model'], r['cut_layer']))
        for column, delta in DELTA_COLUMNS:
            r[delta] = r[column] - base[column] if base is not None else float('nan')
        r['vulnerability'] = vulnerability(r['norm_leak_auc_cut'], r['cosine_leak_auc_cut'],
                                           r['cosine_leak_auc_cut_cleanref'])
    return rows


def _signed(value, digits):
    return f"{value:+.{digits}f}"


def print_thesis_tables(rows, model_name, cut_layer):
    base = next((r for r in rows if r['method'] == 'no_noise'), None)
    if base is None:
        return
    rep_row = next((r for r in rows if r['defense'] == REPRESENTATIVE_DEFENSE), None)
    if rep_row is None:
        rep_row = next((r for r in rows if r['method'] != 'no_noise'), None)

    print("\n" + "=" * 96)
    print(f" LABEL LEAKAGE vs. LABEL PROTECTION -- {model_name} | Cut layer {cut_layer} | {Config.DATASET}")
    print("=" * 96)

    print(" 1. METHOD COMPARISON")
    print("-" * 96)
    print(f"{'Method / Setting':<30} {'Accuracy (%)':>13} {'Best Acc (%)':>13} "
          f"{'Norm AUC':>10} {'Cos AUC':>10} {'Norm Rec (%)':>13}")
    print("-" * 96)
    print(f"{'Vanilla SL (No Defense)':<30} {base['test_accuracy_pct']:>13.2f} {base['best_test_accuracy_pct']:>13.2f} "
          f"{base['norm_leak_auc_cut']:>10.4f} {base['cosine_leak_auc_cut']:>10.4f} {base['norm_recovery_acc_pct']:>13.2f}")
    if rep_row is not None:
        print(f"{rep_row['defense'] + ' defense':<30} {rep_row['test_accuracy_pct']:>13.2f} "
              f"{rep_row['best_test_accuracy_pct']:>13.2f} {rep_row['norm_leak_auc_cut']:>10.4f} "
              f"{rep_row['cosine_leak_auc_cut']:>10.4f} {rep_row['norm_recovery_acc_pct']:>13.2f}")
        print("-" * 96)
        print(f"{'Defense Impact (Delta)':<30} {_signed(rep_row['delta_test_accuracy_pct'], 2) + ' %':>13} "
              f"{_signed(rep_row['delta_best_test_accuracy_pct'], 2) + ' %':>13} "
              f"{_signed(rep_row['delta_norm_leak_auc_cut'], 4):>10} {_signed(rep_row['delta_cosine_leak_auc_cut'], 4):>10} "
              f"{_signed(rep_row['delta_norm_recovery_acc_pct'], 2) + ' %':>13}")

    print("\n 2. THE COMPLETE THESIS TABLE (ATTACK VS LABEL PROTECTION DEFENSES)")
    print("-" * 158)
    print(f"{'Defense':<16} {'Acc (%)':>8} {'Δ Acc':>8} {'Best Acc':>9} {'Bal Acc':>8} {'Test AUC':>9} "
          f"{'Norm AUC':>9} {'Δ Norm':>8} {'Cos AUC':>8} {'Δ Cos':>8} {'Cos AUC*':>9} {'Rec (%)':>8} "
          f"{'WC Bal (%)':>11} {'Def ms':>7}  Vulnerability Assessment")
    print("-" * 158)
    for r in rows:
        is_base = r['method'] == 'no_noise'
        print(f"{r['defense']:<16} {r['test_accuracy_pct']:>8.2f} "
              f"{'—' if is_base else _signed(r['delta_test_accuracy_pct'], 2):>8} "
              f"{r['best_test_accuracy_pct']:>9.2f} {r['test_balanced_accuracy_pct']:>8.2f} {r['test_auc']:>9.4f} "
              f"{r['norm_leak_auc_cut']:>9.4f} {'—' if is_base else _signed(r['delta_norm_leak_auc_cut'], 4):>8} "
              f"{r['cosine_leak_auc_cut']:>8.4f} {'—' if is_base else _signed(r['delta_cosine_leak_auc_cut'], 4):>8} "
              f"{r['cosine_leak_auc_cut_cleanref']:>9.4f} {r['norm_recovery_acc_pct']:>8.2f} "
              f"{r['norm_worstcase_bal_acc_pct']:>11.2f} {r['defense_ms_per_batch']:>7.3f}  {r['vulnerability']}")
    print("-" * 158)
    print(f"  Chance level: attacker accuracy {base['chance_accuracy_pct']:.2f}% (predict all negative), "
          f"leak AUC 0.5, worst-case balanced accuracy 50%.")
    print("  Acc = main-task test accuracy (final epoch) | Rec = label-free norm attack accuracy | "
          "WC Bal = worst-case attacker balanced accuracy")
    print("  Cos AUC = cosine attack as in label_leakage_attack.py (reference from the received batch) | "
          "Cos AUC* = clean positive reference (official code)")

    if rep_row is not None:
        print("\n  Interpretation:")
        print(f"  {rep_row['defense']} moved the strongest leak AUC from "
              f"{max(base['norm_leak_auc_cut'], base['cosine_leak_auc_cut'], base['cosine_leak_auc_cut_cleanref']):.4f} to "
              f"{max(rep_row['norm_leak_auc_cut'], rep_row['cosine_leak_auc_cut'], rep_row['cosine_leak_auc_cut_cleanref']):.4f} "
              f"({rep_row['vulnerability']}) at a main-task accuracy change of "
              f"{rep_row['delta_test_accuracy_pct']:+.2f} points.")
    print("=" * 96)


def print_table(results, title):
    print("\n" + "=" * 120)
    print(f"   {title} -- {Config.DATASET}")
    print("=" * 120)
    print(f"  {'Model':<11} {'Cut':>5} {'Defense':<16} {'NormAUC':>8} {'CosAUC':>8} {'CosAUC*':>8} "
          f"{'NormRec%':>9} {'NormWC%':>8} {'Chance%':>8} {'TestAcc%':>9} {'TestAUC':>8} {'Def ms':>8}")
    print("-" * 120)
    for r in results:
        print(f"  {r['model']:<11} {str(r['cut_layer']):>5} {r['defense']:<16} "
              f"{r['norm_leak_auc_cut']:>8.4f} {r['cosine_leak_auc_cut']:>8.4f} {r['cosine_leak_auc_cut_cleanref']:>8.4f} "
              f"{r['norm_recovery_acc_pct']:>9.2f} {r['norm_worstcase_bal_acc_pct']:>8.2f} {r['chance_accuracy_pct']:>8.2f} "
              f"{r['test_accuracy_pct']:>9.2f} {r['test_auc']:>8.4f} {r['defense_ms_per_batch']:>8.3f}")
    print("=" * 120)


def cut_layer_spread(df):
    metrics = ["norm_leak_auc_cut", "cosine_leak_auc_cut", "cosine_leak_auc_cut_cleanref",
               "norm_worstcase_bal_acc_pct", "cosine_worstcase_bal_acc_pct",
               "test_accuracy_pct", "defense_ms_per_batch"]
    rows = []
    for (model, defense), group in df.groupby(["model", "defense"]):
        row = {"model": model, "defense": defense, "num_cut_layers": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
            row[f"{metric}_range"] = float(np.max(values) - np.min(values))
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, _ = dataset.get_loaders()
    base_dataset = train_loader.dataset

    in_channels = 1 if Config.DATASET == 'MNIST' else 3

    loaders = build_binary_split_loaders(
        base_dataset,
        target_class=POSITIVE_CLASS,
        positive_ratio=POSITIVE_RATIO,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    if RUN_PHASE_A_DEFENSE_SWEEP:
        phase_a = []
        for model_name in MODELS_TO_RUN:
            model_rows = [run_single(model_name, Config.CUT_LAYER, method, params,
                                     in_channels, loaders, "PHASE A")
                          for method, params in DEFENSES_TO_RUN]
            phase_a.extend(model_rows)

        add_thesis_columns(phase_a)
        path_a = f"{Config.RESULTS_DIR}/label_protection_phaseA_defenses_cut{Config.CUT_LAYER}_{Config.DATASET}.csv"
        pd.DataFrame(phase_a).to_csv(path_a, index=False)
        print_table(phase_a, f"LABEL PROTECTION PHASE A: DEFENSE SWEEP AT CUT LAYER {Config.CUT_LAYER}")
        for model_name in MODELS_TO_RUN:
            model_rows = [r for r in phase_a if r['model'] == model_name]
            print_thesis_tables(model_rows, model_name, model_rows[0]['cut_layer'])
        print(f"  Saved: {path_a}")

    if RUN_PHASE_B_CUT_LAYER_SWEEP:
        phase_b = []
        for model_name in CUT_LAYER_SWEEP_MODELS:
            model_rows = [run_single(model_name, cut_layer, method, params,
                                     in_channels, loaders, "PHASE B")
                          for cut_layer in CUT_LAYERS_TO_SWEEP
                          for method, params in CUT_LAYER_SWEEP_DEFENSES]
            phase_b.extend(model_rows)

        add_thesis_columns(phase_b)
        df_b = pd.DataFrame(phase_b)
        path_b = f"{Config.RESULTS_DIR}/label_protection_phaseB_cut_layers_{Config.DATASET}.csv"
        df_b.to_csv(path_b, index=False)
        spread = cut_layer_spread(df_b)
        path_spread = f"{Config.RESULTS_DIR}/label_protection_phaseB_cut_layer_spread_{Config.DATASET}.csv"
        spread.to_csv(path_spread, index=False)

        print_table(phase_b, "LABEL PROTECTION PHASE B: CUT-LAYER SWEEP")
        for model_name in CUT_LAYER_SWEEP_MODELS:
            for cut_layer in CUT_LAYERS_TO_SWEEP:
                rows = [r for r in phase_b if r['model'] == model_name and r['cut_layer'] == cut_layer]
                if rows:
                    print_thesis_tables(rows, model_name, cut_layer)
        print("\n  Spread across cut layers (std of norm leak AUC / worst-case bal. acc %):")
        for _, r in spread.iterrows():
            print(f"    {r['model']:<11} {r['defense']:<16} "
                  f"{r['norm_leak_auc_cut_std']:.4f} / {r['norm_worstcase_bal_acc_pct_std']:.2f}")
        print(f"  Saved: {path_b}")
        print(f"  Saved: {path_spread}")