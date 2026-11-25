import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import random
import traceback

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
from all_defences.label_protection_defense import (LabelProtectionDefense, attach_label_protection,
                                                   LabelAwareCriterion)


SEED = 0

DEFENSES_TO_RUN = (
    [("no_noise", {})]
    + [("max_norm", {})]
    + [("iso", {"ratio": r}) for r in [4.5, 6.0, 9.0, 11.0, 13.0, 15.0]]
    + [("marvell", {"init_scale": s, "p_frac": "pos_frac", "uv_choice": "uv", "dynamic": False})
       for s in [0.05, 0.15, 0.25, 0.4, 1.5, 1.75]]
    + [("perp", {"lower": 1.0, "upper": 5.0})]
)

ATTACKS_TO_RUN = ["label_leakage", "villain", "poison_client", "poison_server",
                  "whitebox", "unsplit", "ae_decoder", "fsha"]

TRAINING_EPOCHS = 5
POSITIVE_CLASS  = 0
POSITIVE_RATIO  = 0.1
BATCH_SIZE      = 128
LEARNING_RATE   = 1e-4

VILLAIN_WARMUP     = 5
VILLAIN_INFERENCE  = 5
VILLAIN_INJECTION  = 10
VILLAIN_BATCH      = 128
VILLAIN_TARGET     = 0
VILLAIN_POISON     = 0.01
VILLAIN_CANDIDATES = 14

POISON_TARGET       = 0
POISON_RATE         = 0.05
POISON_PATCH        = 4
POISON_TRIGGER      = 1.0
POISON_TRAIN_EPOCHS = 5

QUICK_TEST = False

REPRESENTATIVE_DEFENSE = "marvell_s1.75"

RUN_PHASE_A_DEFENSE_SWEEP = True
RUN_PHASE_B_CUT_LAYER_SWEEP = True

CUT_LAYER_SWEEP_MODELS = ["PyramidCNN", "KAGN"]
CUT_LAYERS_TO_SWEEP    = [1, 2, 3, 4, 5]
CUT_LAYER_SWEEP_DEFENSES = [
    ("no_noise", {}),
    ("max_norm", {}),
    ("iso", {"ratio": 9.0}),
    ("marvell", {"init_scale": 0.25, "p_frac": "pos_frac", "uv_choice": "uv", "dynamic": False}),
    ("marvell", {"init_scale": 1.75, "p_frac": "pos_frac", "uv_choice": "uv", "dynamic": False}),
]

VULNERABILITY_LEVELS = [
    (0.95, "Critical Breach (near-exact label recovery)"),
    (0.80, "High Breach"),
    (0.65, "Moderate Breach"),
    (0.55, "Weak / Degraded"),
    (0.00, "Protected (near chance)"),
]

ATTACKS = {
    "label_leakage": ("Label Leakage (norm + cosine gradient scoring, Li et al. ICLR 2022)", "label_inference"),
    "villain":       ("VILLAIN backdoor (label inference + embedding trigger)", "backdoor"),
    "poison_client": ("Backdoor poisoning, client attacker (BadNets patch)", "backdoor"),
    "poison_server": ("Backdoor poisoning, server attacker (surrogate client)", "backdoor"),
    "whitebox":      ("White-box model inversion", "reconstruction"),
    "unsplit":       ("UnSplit model inversion", "reconstruction"),
    "ae_decoder":    ("AE decoder inversion", "reconstruction"),
    "fsha":          ("FSHA feature-space hijacking", "reconstruction"),
}

NOT_APPLICABLE_REASON = {
    "poison_server": "server is the attacker; label protection runs on the server itself",
    "whitebox":      "inference-time inversion of smashed data; no gradient to perturb",
    "unsplit":       "inference-time inversion of smashed data; no gradient to perturb",
    "ae_decoder":    "inference-time inversion of smashed data; no gradient to perturb",
    "fsha":          "malicious server crafts the gradients itself; server-side gradient noise does not apply",
}

DEFENSE_STAGE = "training: perturbs server->client cut-layer gradients every batch"


SUCCESS, PARTIAL, FAILED, NOT_APPLICABLE, WEAK, BASELINE = \
    "SUCCESS", "PARTIAL", "FAILED", "N/A", "ATTACK-WEAK", "BASELINE"

THRESHOLDS = {
    'leak_auc_success': 0.65,
    'leak_auc_partial_drop': 0.10,
    'asr_success': 20.0,
    'asr_partial_drop': 25.0,
    'ssim_success': 0.30,
    'ssim_partial_drop': 0.10,
    'psnr_partial_drop': 3.0,
    'utility_tolerance_pct': 5.0,
}

SUMMARY_COLUMNS = [
    'dataset', 'model', 'cut_layer', 'attack', 'attack_name', 'attack_family', 'defense',
    'attack_epochs', 'defense_epochs', 'defense_stage', 'attack_iterations', 'iteration_unit', 'defense_calls',
    'accuracy_no_defense', 'accuracy_with_defense', 'delta_accuracy',
    'metric', 'metric_no_defense', 'metric_with_defense',
    'verdict', 'defense_working', 'reason',
]


def _missing(v):
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def judge(family, base, dfd):
    if _missing(base) or _missing(dfd):
        return NOT_APPLICABLE, "metric unavailable"
    t = THRESHOLDS
    if family == "label_inference":
        thr, drop, name, fmt = t['leak_auc_success'], t['leak_auc_partial_drop'], "leak AUC", ".3f"
    elif family == "backdoor":
        thr, drop, name, fmt = t['asr_success'], t['asr_partial_drop'], "ASR %", ".1f"
    else:
        thr, drop, name, fmt = t['ssim_success'], t['ssim_partial_drop'], "SSIM", ".3f"
    change = f"{name} {base:{fmt}} -> {dfd:{fmt}}"
    if base < thr:
        return WEAK, f"attack already weak without defense ({name} {base:{fmt}} < {thr})"
    if dfd < thr:
        return SUCCESS, f"{change} (below {thr})"
    if base - dfd >= drop:
        return PARTIAL, f"{change} (reduced, still >= {thr})"
    return FAILED, f"{change} (attack still works)"


def finalize_row(row):
    a0, a1 = row.get('accuracy_no_defense'), row.get('accuracy_with_defense')
    row['delta_accuracy'] = float(a1) - float(a0) if not (_missing(a0) or _missing(a1)) else float('nan')
    row.setdefault('metric', {"label_inference": "LeakAUC", "backdoor": "ASR%",
                              "reconstruction": "SSIM"}[row['attack_family']])
    if row.get('verdict') in (NOT_APPLICABLE, BASELINE):
        row['defense_working'] = "N/A" if row['verdict'] == NOT_APPLICABLE else "-"
        row.setdefault('reason', "")
        return row
    verdict, reason = judge(row['attack_family'], row.get('metric_no_defense'), row.get('metric_with_defense'))
    utility_ok = True
    if not _missing(row['delta_accuracy']):
        utility_ok = row['delta_accuracy'] >= -THRESHOLDS['utility_tolerance_pct']
        reason += f"; accuracy {row['delta_accuracy']:+.2f} pts"
    row['verdict'] = verdict
    row['defense_working'] = {SUCCESS: "YES" if utility_ok else "YES (high acc. cost)",
                              PARTIAL: "PARTIAL", FAILED: "NO"}.get(verdict, verdict)
    row['reason'] = reason
    return row


def _fmt(v, spec, width):
    if _missing(v):
        return f"{'-':>{width}}"
    if isinstance(v, str):
        return f"{v:>{width}}"
    return f"{format(v, spec):>{width}}"


def print_summary_table(rows, title):
    line = "=" * 178
    print("\n" + line)
    print(f"   SUMMARY -- {title}")
    print(line)
    print(f"  {'Attack':<14} {'Defense':<16} {'Cut':>5} {'AtkEp':>6} {'DefEp':>6} {'Iterations':>11} "
          f"{'DefCalls':>9} {'Acc0%':>7} {'Acc%':>7} {'dAcc':>7} {'Metric':>8} {'Before':>8} {'After':>8}  "
          f"{'Verdict':<12} {'Working':<20}")
    print("-" * 178)
    for r in rows:
        spec = '.2f' if r['attack_family'] == 'backdoor' else '.4f'
        print(f"  {r['attack']:<14} {r['defense']:<16.16} {str(r['cut_layer']):>5} "
              f"{_fmt(r.get('attack_epochs'), '.0f', 6)} {_fmt(r.get('defense_epochs'), '.0f', 6)} "
              f"{_fmt(r.get('attack_iterations'), ',.0f', 11)} {_fmt(r.get('defense_calls'), ',.0f', 9)} "
              f"{_fmt(r.get('accuracy_no_defense'), '.2f', 7)} {_fmt(r.get('accuracy_with_defense'), '.2f', 7)} "
              f"{_fmt(r.get('delta_accuracy'), '+.2f', 7)} {r.get('metric', ''):>8} "
              f"{_fmt(r.get('metric_no_defense'), spec, 8)} {_fmt(r.get('metric_with_defense'), spec, 8)}  "
              f"{r['verdict']:<12} {r['defense_working']:<20}")
    print("-" * 178)
    print("  AtkEp = attack training epochs | DefEp = epochs the defense was active | Iterations = attack training "
          "steps | DefCalls = batches the defense perturbed")
    print("  Acc0/Acc = main-task accuracy without/with defense | Before/After = attack metric without/with defense "
          f"| SUCCESS if LeakAUC < {THRESHOLDS['leak_auc_success']}, ASR < {THRESHOLDS['asr_success']}%, "
          f"SSIM < {THRESHOLDS['ssim_success']}")
    print(f"  Accuracy loss > {THRESHOLDS['utility_tolerance_pct']} pts is flagged as high cost | "
          "ATTACK-WEAK = attack failed even without defense, so the defense cannot be judged")
    print(line)


def print_defense_attack_matrix(rows):
    rows = [r for r in rows if r['verdict'] != BASELINE]
    if not rows:
        return
    attacks = list(dict.fromkeys(r['attack'] for r in rows))
    defenses = list(dict.fromkeys(r['defense'] for r in rows))
    cell = {(r['defense'], r['attack']): r['defense_working'] for r in rows}
    width = max(15, max(len(a) for a in attacks) + 2)
    print("\n" + "=" * (20 + width * len(attacks)))
    print("   WHICH DEFENSE WORKS AGAINST WHICH ATTACK")
    print("=" * (20 + width * len(attacks)))
    print(f"  {'Defense':<16}" + "".join(f"{a:>{width}}" for a in attacks))
    print("-" * (20 + width * len(attacks)))
    for d in defenses:
        print(f"  {d:<16.16}" + "".join(f"{cell.get((d, a), '.')[:width - 2]:>{width}}" for a in attacks))
    print("=" * (20 + width * len(attacks)))
    for d in defenses:
        mine = [r for r in rows if r['defense'] == d]
        works = [r['attack'] for r in mine if r['verdict'] == SUCCESS]
        partial = [r['attack'] for r in mine if r['verdict'] == PARTIAL]
        fails = [r['attack'] for r in mine if r['verdict'] == FAILED]
        print(f"  {d:<16} successful against: {', '.join(works) or '-'} | partial: {', '.join(partial) or '-'} | "
              f"fails: {', '.join(fails) or '-'}")


def save_results(rows, path):
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
        df['cut_layer'] = df['cut_layer'].astype(str)
        df = df.drop_duplicates(subset=['dataset', 'model', 'cut_layer', 'attack', 'defense'], keep='last')
    df = df[[c for c in SUMMARY_COLUMNS if c in df.columns] + [c for c in df.columns if c not in SUMMARY_COLUMNS]]
    df.to_csv(path, index=False)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_from_config():
    name = Config.MODEL_NAME.lower()
    if "kagn" in name:
        return "KAGN"
    if "pyramid" in name:
        return "PyramidCNN"
    return "Vanilla"


def build_split_models(model_name, in_channels, cut_layer, num_classes=1):
    if model_name == "KAGN":
        client = KAGNClientModel(cut_layer=cut_layer, in_channels=in_channels, degree=Config.DEGREE)
        server = KAGNServerModel(cut_layer=cut_layer, num_classes=num_classes, in_channels=in_channels,
                                 degree=Config.DEGREE)
    elif model_name == "PyramidCNN":
        client = PyramidCNNClientModel(cut_layer=cut_layer, in_channels=in_channels)
        server = PyramidCNNServerModel(cut_layer=cut_layer, num_classes=num_classes, in_channels=in_channels)
    else:
        client = ClientModel(in_channels=in_channels)
        server = ServerModel(num_classes=num_classes)
    return client, server


def clean_checkpoint_path(model_name):
    tag = {"Vanilla": "vanilla_sl", "PyramidCNN": "pyramidcnn_sl", "KAGN": "kagn_sl"}[model_name]
    for path in (f"{Config.SAVE_DIR}/best_{tag}_{Config.DATASET}.pth",
                 f"{Config.SAVE_DIR}/best_{Config.MODEL_NAME.lower()}_sl_{Config.DATASET}.pth"):
        if os.path.exists(path):
            return path
    return f"{Config.SAVE_DIR}/best_{tag}_{Config.DATASET}.pth"


def attach_with_label_map(attack, defense, label_map):
    protected = attach_label_protection(attack, defense)

    class MappedCriterion(LabelAwareCriterion):
        def forward(self, logits, labels):
            self.protected_server.set_labels(label_map(labels))
            return self.criterion(logits, labels)

    attack.criterion = MappedCriterion(attack.criterion.criterion, protected)
    return protected


def base_row(attack_key, model_name, cut_label, defense_tag):
    name, family = ATTACKS[attack_key]
    return {"dataset": Config.DATASET, "model": model_name, "cut_layer": cut_label, "attack": attack_key,
            "attack_name": name, "attack_family": family, "defense": defense_tag}


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


def run_single(model_name, cut_layer, cut_label, method, params, in_channels, loaders, phase="PHASE A"):
    set_seed()
    defense = LabelProtectionDefense(method, **params)
    tag = defense.tag()

    print("\n" + "#" * 78)
    print(f"#   {phase} | MODEL: {model_name} | CUT LAYER: {cut_label} | DEFENSE: {tag} | {Config.DATASET}")
    print("#" * 78)

    client_model, server_model = build_split_models(model_name, in_channels, cut_layer, num_classes=1)
    attack = TrackedLabelLeakageAttack(client_model=client_model, server_model=server_model,
                                       dataset=Config.DATASET, target_class=POSITIVE_CLASS,
                                       learning_rate=LEARNING_RATE)
    observer = GradientObserver()
    protected = attach_label_protection(attack, defense, observer)
    summary = attack.run(loaders[0], loaders[1], epochs=TRAINING_EPOCHS)

    run_tag = f"{model_name.lower()}_cut{cut_label}_{tag}_{Config.DATASET}"
    pd.DataFrame(attack.history).to_csv(f"{Config.RESULTS_DIR}/label_protection_batches_{run_tag}.csv", index=False)
    pd.DataFrame(observer.history).to_csv(f"{Config.RESULTS_DIR}/label_protection_recovery_{run_tag}.csv", index=False)
    pd.DataFrame(attack.epoch_history).to_csv(f"{Config.RESULTS_DIR}/label_protection_epochs_{run_tag}.csv",
                                              index=False)
    if defense.method == "marvell":
        pd.DataFrame(defense.solver_log).to_csv(
            f"{Config.RESULTS_DIR}/label_protection_marvell_solver_{run_tag}.csv", index=False)

    extra = observer.summary()
    epochs = attack.epoch_history
    solver_ms = [entry['solver_ms'] for entry in defense.solver_log]
    is_baseline = defense.method == "no_noise"

    row = base_row("label_leakage", model_name, cut_label, tag)
    row.update({
        "phase": phase,
        "method": defense.method,
        "attack_epochs": TRAINING_EPOCHS,
        "defense_epochs": 0 if is_baseline else TRAINING_EPOCHS,
        "defense_stage": "none" if is_baseline else DEFENSE_STAGE,
        "attack_iterations": len(loaders[0]) * TRAINING_EPOCHS,
        "iteration_unit": "training steps (train batches x epochs)",
        "defense_calls": 0 if is_baseline else len(protected.defense_ms),
        "smashed_dim": int(np.prod(attack.smashed_shape[1:])),
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
    })
    row["leak_auc"] = float(np.nanmax([row["norm_leak_auc_cut"], row["cosine_leak_auc_cut"]]))
    return row


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


def add_verdict_columns(rows):
    base = next((r for r in rows if r['method'] == 'no_noise'), None)
    for r in rows:
        r['accuracy_with_defense'] = r['test_accuracy_pct']
        r['metric_with_defense'] = r['leak_auc']
        r['accuracy_no_defense'] = base['test_accuracy_pct'] if base else float('nan')
        r['metric_no_defense'] = base['leak_auc'] if base else float('nan')
        if r['method'] == 'no_noise':
            r['verdict'] = BASELINE
        finalize_row(r)
    return rows


def update_phase_b(rows, model_name):
    sweep_tags = {LabelProtectionDefense(m, **{**p, **({'verbose': False} if m == 'marvell' else {})}).tag()
                  for m, p in CUT_LAYER_SWEEP_DEFENSES}
    new = pd.DataFrame([{**r, "phase": "PHASE B"} for r in rows if r['defense'] in sweep_tags])
    path_b = f"{Config.RESULTS_DIR}/label_protection_phaseB_cut_layers_{Config.DATASET}.csv"
    if os.path.exists(path_b):
        df_b = pd.concat([pd.read_csv(path_b), new], ignore_index=True)
        df_b = df_b.drop_duplicates(subset=["model", "cut_layer", "defense"], keep="last")
    else:
        df_b = new
    df_b = df_b.sort_values(["model", "cut_layer"])
    df_b.to_csv(path_b, index=False)
    print(f"  Saved: {path_b}")

    done = sorted(int(c) for c in df_b[df_b['model'] == model_name]['cut_layer'].unique())
    missing = [c for c in CUT_LAYERS_TO_SWEEP if c not in done]
    print(f"\n  Phase B cut-layer sweep for {model_name}: cut layers collected {done}"
          + (f", still missing {missing} (set Config.CUT_LAYER and rerun)" if missing else ", complete"))
    if len(done) < 2:
        return
    phase_b_rows = df_b.to_dict("records")
    print_table(phase_b_rows, "LABEL PROTECTION PHASE B: CUT-LAYER SWEEP")
    for cut in done:
        cut_rows = [r for r in phase_b_rows if r['model'] == model_name and int(r['cut_layer']) == cut]
        if cut_rows:
            print_thesis_tables(cut_rows, model_name, cut)
    spread = cut_layer_spread(df_b)
    path_spread = f"{Config.RESULTS_DIR}/label_protection_phaseB_cut_layer_spread_{Config.DATASET}.csv"
    spread.to_csv(path_spread, index=False)
    print("\n  Spread across cut layers (std of norm leak AUC / worst-case bal. acc %):")
    for _, r in spread.iterrows():
        print(f"    {r['model']:<11} {r['defense']:<16} "
              f"{r['norm_leak_auc_cut_std']:.4f} / {r['norm_worstcase_bal_acc_pct_std']:.2f}")
    print(f"  Saved: {path_spread}")


def run_villain(model_name, cut_layer, cut_label, method, params, in_channels, base_dataset, test_loader):
    from all_attacks.villain_backdoor_attack import VILLAINBackdoorAttack, build_indexed_loader
    set_seed()
    defense = LabelProtectionDefense(method, **params)
    client, server = build_split_models(model_name, in_channels, cut_layer, Config.NUM_CLASSES)
    attack = VILLAINBackdoorAttack(client_model=client, server_model=server, base_dataset=base_dataset,
                                   dataset=Config.DATASET, num_classes=Config.NUM_CLASSES,
                                   target_label=VILLAIN_TARGET, poison_rate=VILLAIN_POISON,
                                   candidates_per_batch=VILLAIN_CANDIDATES)
    protected = None
    if defense.method != "no_noise":
        protected = attach_with_label_map(attack, defense, lambda y: (y == VILLAIN_TARGET).float())

    loader = build_indexed_loader(base_dataset, batch_size=VILLAIN_BATCH, shuffle=True)
    attack.warmup(loader, epochs=VILLAIN_WARMUP)
    attack.infer_labels(loader, epochs=VILLAIN_INFERENCE)
    attack.fabricate_trigger(loader)
    attack.inject_backdoor(loader, test_loader, epochs=VILLAIN_INJECTION)
    cda, asr = attack.evaluate(test_loader)
    total_epochs = VILLAIN_WARMUP + VILLAIN_INFERENCE + VILLAIN_INJECTION

    row = base_row("villain", model_name, cut_label, defense.tag())
    row.update({"method": defense.method, "attack_epochs": total_epochs,
                "attack_epochs_detail": f"warmup {VILLAIN_WARMUP} + inference {VILLAIN_INFERENCE} + "
                                        f"injection {VILLAIN_INJECTION}",
                "defense_epochs": 0 if protected is None else total_epochs,
                "defense_stage": "none" if protected is None else DEFENSE_STAGE + " (all VILLAIN phases)",
                "attack_iterations": len(loader) * total_epochs,
                "iteration_unit": "training steps (train batches x all VILLAIN epochs)",
                "defense_calls": 0 if protected is None else len(protected.defense_ms),
                "accuracy_with_defense": cda, "metric_with_defense": asr,
                "lia": 100.0 * attack.label_inference_accuracy()})
    return row


def run_poison_client(model_name, cut_layer, cut_label, method, params, in_channels, base_dataset,
                      train_loader, test_loader):
    from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack
    set_seed()
    defense = LabelProtectionDefense(method, **params)
    client, server = build_split_models(model_name, in_channels, cut_layer, Config.NUM_CLASSES)
    tag = {"Vanilla": "vanilla_sl", "PyramidCNN": "pyramidcnn_sl", "KAGN": "kagn_sl"}[model_name]
    attack = BackdoorPoisonAttack(client_model=client, server_model=server, base_dataset=base_dataset,
                                  dataset=Config.DATASET, num_classes=Config.NUM_CLASSES, mode="client",
                                  target_label=POISON_TARGET, poison_rate=POISON_RATE, patch_size=POISON_PATCH,
                                  trigger_value=POISON_TRIGGER, model_tag=tag)
    attack._save_checkpoint = lambda *a, **k: None
    attack.load_clean_init(clean_checkpoint_path(model_name))
    protected = None
    if defense.method != "no_noise":
        protected = attach_with_label_map(attack, defense, lambda y: (y == POISON_TARGET).float())
    attack.train(train_loader, test_loader, epochs=POISON_TRAIN_EPOCHS)
    cda, asr = attack.evaluate(test_loader)

    row = base_row("poison_client", model_name, cut_label, defense.tag())
    row.update({"method": defense.method, "attack_epochs": POISON_TRAIN_EPOCHS,
                "defense_epochs": 0 if protected is None else POISON_TRAIN_EPOCHS,
                "defense_stage": "none" if protected is None else DEFENSE_STAGE + " (poisoned training)",
                "attack_iterations": len(train_loader) * POISON_TRAIN_EPOCHS,
                "iteration_unit": "training steps (train batches x epochs)",
                "defense_calls": 0 if protected is None else len(protected.defense_ms),
                "accuracy_with_defense": cda, "metric_with_defense": asr})
    return row


def finish_backdoor_rows(rows):
    base = next((r for r in rows if r['method'] == 'no_noise'), None)
    for r in rows:
        r['accuracy_no_defense'] = base['accuracy_with_defense'] if base else float('nan')
        r['metric_no_defense'] = base['metric_with_defense'] if base else float('nan')
        if r['method'] == 'no_noise':
            r['verdict'] = BASELINE
        finalize_row(r)
        if 'lia' in r and base is not None and r['verdict'] != BASELINE:
            r['reason'] += f"; label inference {base['lia']:.1f}% -> {r['lia']:.1f}%"
    return rows


if __name__ == "__main__":
    if QUICK_TEST:
        TRAINING_EPOCHS = VILLAIN_WARMUP = VILLAIN_INFERENCE = VILLAIN_INJECTION = POISON_TRAIN_EPOCHS = 1

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    model_name = model_from_config()
    cut_layer = Config.CUT_LAYER
    cut_label = "fixed" if model_name == "Vanilla" else cut_layer
    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    defense_tags = [LabelProtectionDefense(m, **{**p, **({'verbose': False} if m == 'marvell' else {})}).tag()
                    for m, p in DEFENSES_TO_RUN]

    print("=" * 78)
    print("  LABEL PROTECTION DEFENSES vs ALL ATTACKS")
    print(f"  Dataset  : {Config.DATASET}")
    print(f"  Model    : {model_name}  (Config.MODEL_NAME = {Config.MODEL_NAME})")
    print(f"  Cut layer: {cut_label}")
    print(f"  Device   : {device}")
    print(f"  Defenses : {defense_tags}")
    print(f"  Attacks  : {ATTACKS_TO_RUN}")
    print("=" * 78)

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()
    base_dataset = train_loader.dataset

    all_rows = []
    output_path = f"{Config.RESULTS_DIR}/label_protection_all_attacks_{Config.DATASET}.csv"

    for attack_key in ATTACKS_TO_RUN:
        print("\n" + "#" * 78)
        print(f"#   ATTACK: {attack_key} -- {ATTACKS[attack_key][0]}")
        print(f"#   MODEL: {model_name} | CUT LAYER: {cut_label} | {Config.DATASET}")
        print("#" * 78)

        if attack_key in NOT_APPLICABLE_REASON:
            for (method, params), tag in zip(DEFENSES_TO_RUN, defense_tags):
                if method == "no_noise":
                    continue
                row = base_row(attack_key, model_name, cut_label, tag)
                row.update({"verdict": NOT_APPLICABLE, "reason": NOT_APPLICABLE_REASON[attack_key],
                            "defense_epochs": 0, "defense_calls": 0})
                all_rows.append(finalize_row(row))
            print(f"  [N/A] {NOT_APPLICABLE_REASON[attack_key]}")
            continue

        attack_rows = []
        for method, params in DEFENSES_TO_RUN:
            print(f"\n  >> {attack_key} | defense: {method} {params}")
            try:
                if attack_key == "label_leakage":
                    if not hasattr(run_single, "loaders"):
                        run_single.loaders = build_binary_split_loaders(
                            base_dataset, target_class=POSITIVE_CLASS, positive_ratio=POSITIVE_RATIO,
                            batch_size=BATCH_SIZE, seed=SEED)
                    row = run_single(model_name, cut_layer, cut_label, method, params, in_channels,
                                            run_single.loaders)
                elif attack_key == "villain":
                    row = run_villain(model_name, cut_layer, cut_label, method, params, in_channels,
                                      base_dataset, test_loader)
                else:
                    row = run_poison_client(model_name, cut_layer, cut_label, method, params, in_channels,
                                            base_dataset, train_loader, test_loader)
            except Exception as exc:
                traceback.print_exc()
                row = base_row(attack_key, model_name, cut_label, method)
                row.update({"method": method, "verdict": NOT_APPLICABLE,
                            "reason": f"run failed: {type(exc).__name__}: {exc}"})
            attack_rows.append(row)
            print(f"  [run info] attack={attack_key} | cut={cut_label} | attack epochs={row.get('attack_epochs')} | "
                  f"defense epochs={row.get('defense_epochs')} | iterations={row.get('attack_iterations')} | "
                  f"defense calls={row.get('defense_calls')}")

        ok_rows = [r for r in attack_rows if r.get('verdict') != NOT_APPLICABLE]
        failed_rows = [finalize_row(r) for r in attack_rows if r.get('verdict') == NOT_APPLICABLE]
        if attack_key == "label_leakage":
            add_thesis_columns(ok_rows)
            add_verdict_columns(ok_rows)
            if RUN_PHASE_A_DEFENSE_SWEEP:
                path_a = f"{Config.RESULTS_DIR}/label_protection_phaseA_defenses_cut{Config.CUT_LAYER}_{Config.DATASET}.csv"
                pd.DataFrame(ok_rows).to_csv(path_a, index=False)
                print_table(ok_rows, f"LABEL PROTECTION PHASE A: DEFENSE SWEEP AT CUT LAYER {Config.CUT_LAYER}")
                print_thesis_tables(ok_rows, model_name, cut_label)
                print(f"  Saved: {path_a}")
            if RUN_PHASE_B_CUT_LAYER_SWEEP and model_name in CUT_LAYER_SWEEP_MODELS:
                update_phase_b(ok_rows, model_name)
        else:
            finish_backdoor_rows(ok_rows)
        for r in ok_rows:
            if r['verdict'] != BASELINE:
                print(f"    {r['defense']:<16} {r['defense_working']:<20} {r['reason']}")
        all_rows.extend(ok_rows + failed_rows)
        save_results(all_rows, output_path)

    print_summary_table(all_rows, f"LABEL PROTECTION vs ALL ATTACKS -- {model_name} | cut {cut_label} | "
                                  f"{Config.DATASET}")
    print_defense_attack_matrix(all_rows)
    save_results(all_rows, output_path)
    print(f"\n  Saved -> {output_path}")