import sys, os
_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_r, 'config.py')) and os.path.dirname(_r) != _r:
    _r = os.path.dirname(_r)
sys.path.append(_r)

import torch
import pandas as pd
from config import Config
from dataset import DatasetLoader
from existing_defense.r3elu_defense import R3eLUMechanism
from all_split_learning.r3elu_split_learning import R3eLUSplitLearningTrainer
from all_attacks.ae_decoder_attack import run_ae_decoder_attack
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel
from all_attacks.attacks_whitebox import WhiteBoxInversionAttack, AttackMetricsTracker
from all_attacks.fsha_attack import FSHAAttack, gradient_penalty, compute_psnr, compute_ssim, compute_mse, \
    denormalize as fsha_denormalize
from all_attacks.fora_attack import FORAAttack
from all_attacks.similarity_label_attack import SimilarityLabelInferenceAttack
from torch.utils.data import DataLoader
import argparse
from defence_runner.new_defense_runner import defended_eval as DE
from all_attacks.pcat_attack import PCATAttack
from all_attacks.property_inference_attack import MaoPropertyInference, PROPERTIES
from all_attacks.membership_inference_attack import ShokriMembershipInference
from util.split_env import load_raw_datasets
PCAT_VANILLA_ITERS = 5000
PROP_N_AUX = 5000
PROP_N_EVAL = 10000
PROP_EPOCHS = 30
MIA_SHADOWS = 10
MIA_TARGET = 2000
_VALID = None
SIM_EPOCHS = 1
FORA_ITERS = 4000
KEY_TO_VALID = {'whitebox':'whitebox','ae_decoder':'ae_decoder','fsha':'fsha','fora':'fora',
                'label_leakage':'label_leakage','similarity_label':'similarity_label',
                'poison_client':'backdoor_client','poison_server':'backdoor_server',
                'pcat':'pcat','property':'property_inference','membership':'membership'}
def attack_allowed(key):
    if _VALID is None:
        return True
    return (KEY_TO_VALID.get(key, key), Config.DATASET, Config.MODEL_NAME, Config.CUT_LAYER) in _VALID


def build_client(in_channels):
    if Config.MODEL_NAME == "KAGN":
        return KAGNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels, degree=Config.DEGREE)
    if Config.MODEL_NAME == "PyramidCNN":
        return PyramidCNNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels)
    return ClientModel(in_channels=in_channels)


def build_server(in_channels):
    if Config.MODEL_NAME == "KAGN":
        return KAGNServerModel(cut_layer=Config.CUT_LAYER, num_classes=Config.NUM_CLASSES,
                               in_channels=in_channels, degree=Config.DEGREE)
    if Config.MODEL_NAME == "PyramidCNN":
        return PyramidCNNServerModel(cut_layer=Config.CUT_LAYER, num_classes=Config.NUM_CLASSES, in_channels=in_channels)
    return ServerModel(num_classes=Config.NUM_CLASSES)


def vanilla_checkpoint_path():
    return f"{Config.SAVE_DIR}/best_{Config.MODEL_NAME.lower()}_sl_{Config.DATASET}.pth"


def denorm(inputs, dataset):
    if dataset == 'CIFAR10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=inputs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010], device=inputs.device).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.1307], device=inputs.device).view(1, 1, 1, 1)
        std = torch.tensor([0.3081], device=inputs.device).view(1, 1, 1, 1)
    return torch.clamp(inputs * std + mean, 0, 1)


class DefendedClient(nn.Module):

    def __init__(self, client_model, view_fn):
        super().__init__()
        self.client_model = client_model
        self.view_fn = view_fn

    def forward(self, x):
        with torch.no_grad():
            return self.view_fn(self.client_model(x))


def run_whitebox_vs_view(client_model, view_fn, test_loader, device, dataset, max_images=32, iterations=1000):
    attacker = WhiteBoxInversionAttack(client_model=client_model, dataset=dataset, iterations=iterations, lr=1e-2)
    tracker = AttackMetricsTracker()
    done = 0
    client_model.eval()
    for inputs, _ in test_loader:
        if done >= max_images:
            break
        inputs = inputs[:max_images - done].to(device)
        with torch.no_grad():
            observed = view_fn(client_model(inputs))
        recon = attacker.reconstruct(observed, inputs.shape)
        tracker.log_batch(denorm(inputs, dataset), recon)
        done += inputs.shape[0]
    return tracker.get_summary()


class DefendedFSHAAttack(FSHAAttack):

    def __init__(self, *args, view_fn=None, tag='defended', **kw):
        super().__init__(*args, view_fn=view_fn, tag=tag, **kw)


def run_fsha_vs_view(view_fn, in_channels, private_loader, public_loader, tag, epochs=5, num_images=32,
                     client_builder=None, iterations=None):
    builder = client_builder or (lambda: build_client(in_channels))
    attack = DefendedFSHAAttack(client_model=builder(), in_channels=in_channels, dataset=Config.DATASET,
                                pilot_builder=builder, critic_iters=5, view_fn=view_fn, tag=tag)
    attack.hijack(private_loader, public_loader, epochs=epochs, iterations=iterations)
    return attack.reconstruct(private_loader, num_images=num_images)

EPSILON_VALUES   = [0.1, 0.5, 1.0, 2.0, 4.0]
C_CLIP           = 10.0
K_FRACTION       = 0.5
PROTECT_BACKWARD = True
DYNAMIC_BUDGET   = True
EPOCHS_PER_EPS   = 10

RUN_WHITEBOX, RUN_AE, RUN_FSHA, RUN_FORA = True, True, True, True
MAX_IMAGES, ITERATIONS         = 32, 1000
AE_EPOCHS, FSHA_EPOCHS         = 50, 5
RUN_UNSPLIT   = True
UNSPLIT_STEPS = 1000

LEAK_EPOCHS    = 5
LEAK_POS_CLASS = 0
LEAK_POS_RATIO = 0.1
LEAK_BATCH     = 128
LEAK_LR        = 1e-4

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


def run_fora_vs_view(client_model, view_fn, private_loader, public_loader, device, in_channels):
    victim = DefendedClient(client_model, view_fn)
    atk = FORAAttack(victim_client=victim, substitute_builder=lambda: build_client(in_channels),
                     in_channels=in_channels, dataset=Config.DATASET, tag='r3elu')
    atk.run(private_loader, public_loader, iterations=FORA_ITERS)
    s = atk.reconstruct(public_loader, num_images=MAX_IMAGES)
    return {'psnr': s['psnr'], 'ssim': s['ssim']}


def attack_suite(client_model, view_fn, train_loader, test_loader, device, in_channels, tag):
    row = {}
    if RUN_WHITEBOX and attack_allowed('whitebox'):
        s = run_whitebox_vs_view(client_model, view_fn, test_loader, device, Config.DATASET, MAX_IMAGES, ITERATIONS)
        row.update(wb_psnr=s['mean_psnr'], wb_ssim=s['mean_ssim'])
    if RUN_AE and attack_allowed('ae_decoder'):
        s, _, _ = run_ae_decoder_attack(DefendedClient(client_model, view_fn), train_loader, test_loader, device,
                                        Config.DATASET, ae_epochs=AE_EPOCHS, label=f'AE Decoder vs {tag}')
        row.update(ae_psnr=s['mean_psnr'], ae_ssim=s['mean_ssim'])
    if RUN_FSHA and attack_allowed('fsha'):
        s = run_fsha_vs_view(view_fn, in_channels, train_loader, test_loader, tag=tag, epochs=FSHA_EPOCHS)
        row.update(fsha_psnr=s['psnr'], fsha_ssim=s['ssim'])
    if RUN_FORA and attack_allowed('fora'):
        s = run_fora_vs_view(client_model, view_fn, train_loader, test_loader, device, in_channels)
        row.update(fora_psnr=s['psnr'], fora_ssim=s['ssim'])
    return row


import math
import inspect
import random
import traceback

ATTACKS_TO_RUN = ["label_leakage", "similarity_label", "poison_client", "poison_server",
                  "whitebox", "ae_decoder", "fsha", "fora", "pcat", "property", "membership"]

ATTACK_INFO = {
    "label_leakage": ("Label Leakage (norm + cosine gradient scoring)", "label_inference"),
    "similarity_label": ("Similarity label inference (gradient clustering)", "label_cluster"),
    "poison_client": ("Backdoor poisoning, client attacker (BadNets patch)", "backdoor"),
    "poison_server": ("Backdoor poisoning, server attacker (surrogate client)", "backdoor"),
    "whitebox":      ("White-box model inversion", "reconstruction"),
    "ae_decoder":    ("AE decoder inversion", "reconstruction"),
    "fsha":          ("FSHA feature-space hijacking", "reconstruction"),
    "fora":          ("FORA feature-oriented reconstruction", "reconstruction"),
    "pcat":          ("PCAT pseudo-client functionality stealing", "model_stealing"),
    "property":      ("Property inference (Mao et al.)", "property"),
    "membership":    ("Membership inference (Shokri et al.)", "membership"),
}

RECON_KEYS = {"whitebox": "wb", "ae_decoder": "ae", "fsha": "fsha", "fora": "fora"}

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
    'steal_drop': 10.0,
    'prop_success': 55.0,
    'prop_drop': 5.0,
    'mia_success': 0.55,
    'mia_drop': 0.03,
    'cluster_margin': 0.10,
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


def judge(family, base, dfd, psnr_base=None, psnr_dfd=None):
    if _missing(base) or _missing(dfd):
        return NOT_APPLICABLE, "metric unavailable (no undefended baseline to compare with)"
    t = THRESHOLDS
    if family == "label_inference":
        thr, drop, name, fmt = t['leak_auc_success'], t['leak_auc_partial_drop'], "leak AUC", ".3f"
    elif family == "backdoor":
        thr, drop, name, fmt = t['asr_success'], t['asr_partial_drop'], "ASR %", ".1f"
    elif family == "label_cluster":
        base, dfd = float(base), float(dfd)
        chance = 1.0 / float(Config.NUM_CLASSES)
        thr = chance + t['cluster_margin']
        change = f"cluster acc {base:.3f} -> {dfd:.3f} (chance {chance:.2f})"
        if base < thr:
            return WEAK, f"attack already weak without defense (cluster acc {base:.3f} < {thr:.3f})"
        if dfd < thr:
            return SUCCESS, f"{change} (back to chance)"
        if base - dfd >= t['cluster_margin']:
            return PARTIAL, f"{change} (reduced)"
        return FAILED, f"{change} (attack still works)"
    elif family == "model_stealing":
        base, dfd = float(base), float(dfd)
        change = f"stolen acc {base:.1f} -> {dfd:.1f}"
        if base - dfd >= t['steal_drop']:
            return SUCCESS, f"{change} (stolen accuracy cut by >= {t['steal_drop']})"
        if base - dfd >= t['steal_drop'] / 2:
            return PARTIAL, f"{change} (partially reduced)"
        return FAILED, f"{change} (stealing still works)"
    elif family == "property":
        base, dfd = float(base), float(dfd)
        change = f"prop acc {base:.1f} -> {dfd:.1f}"
        if base < t['prop_success']:
            return WEAK, f"attack already weak without defense ({base:.1f} < {t['prop_success']})"
        if dfd < t['prop_success']:
            return SUCCESS, f"{change} (below {t['prop_success']})"
        if base - dfd >= t['prop_drop']:
            return PARTIAL, f"{change} (reduced)"
        return FAILED, f"{change} (attack still works)"
    elif family == "membership":
        base, dfd = float(base), float(dfd)
        change = f"AUC {base:.3f} -> {dfd:.3f}"
        if base < t['mia_success']:
            return WEAK, f"attack already weak without defense (AUC {base:.3f} < {t['mia_success']})"
        if dfd < t['mia_success']:
            return SUCCESS, f"{change} (below {t['mia_success']})"
        if base - dfd >= t['mia_drop']:
            return PARTIAL, f"{change} (reduced)"
        return FAILED, f"{change} (attack still works)"
    else:
        thr, drop, name, fmt = t['ssim_success'], t['ssim_partial_drop'], "SSIM", ".3f"
    base, dfd = float(base), float(dfd)
    change = f"{name} {base:{fmt}} -> {dfd:{fmt}}"
    if base < thr:
        return WEAK, f"attack already weak without defense ({name} {base:{fmt}} < {thr})"
    if dfd < thr:
        return SUCCESS, f"{change} (below {thr})"
    psnr_drop = float(psnr_base) - float(psnr_dfd) if not (_missing(psnr_base) or _missing(psnr_dfd)) else 0.0
    if base - dfd >= drop or (family == "reconstruction" and psnr_drop >= t['psnr_partial_drop']):
        return PARTIAL, f"{change} (reduced, still >= {thr})"
    return FAILED, f"{change} (attack still works)"


def finalize_row(row):
    a0, a1 = row.get('accuracy_no_defense'), row.get('accuracy_with_defense')
    row['delta_accuracy'] = float(a1) - float(a0) if not (_missing(a0) or _missing(a1)) else float('nan')
    row.setdefault('metric', {"label_inference": "LeakAUC", "backdoor": "ASR%", "reconstruction": "SSIM",
                              "model_stealing": "StolenAcc", "property": "PropAcc",
                              "membership": "AUC", "label_cluster": "ClusterAcc"}[row['attack_family']])
    if row.get('verdict') in (NOT_APPLICABLE, BASELINE):
        row['defense_working'] = "N/A" if row['verdict'] == NOT_APPLICABLE else "-"
        row.setdefault('reason', "")
        return row
    verdict, reason = judge(row['attack_family'], row.get('metric_no_defense'), row.get('metric_with_defense'),
                            row.get('psnr_no_defense'), row.get('psnr_with_defense'))
    utility_ok = True
    if not _missing(row['delta_accuracy']):
        utility_ok = row['delta_accuracy'] >= -THRESHOLDS['utility_tolerance_pct']
        reason += f"; accuracy {row['delta_accuracy']:+.2f} pts"
    if row.get('note'):
        reason += f"; {row['note']}"
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
    print(f"  {'Attack':<14} {'Defense':<22} {'Cut':>5} {'AtkEp':>6} {'DefEp':>6} {'Iterations':>11} "
          f"{'DefCalls':>9} {'Acc0%':>7} {'Acc%':>7} {'dAcc':>7} {'Metric':>8} {'Before':>8} {'After':>8}  "
          f"{'Verdict':<12} {'Working':<20}")
    print("-" * 178)
    for r in rows:
        spec = '.2f' if r['attack_family'] == 'backdoor' else '.4f'
        print(f"  {r['attack']:<14} {r['defense']:<22.22} {str(r['cut_layer']):>5} "
              f"{_fmt(r.get('attack_epochs'), '.0f', 6)} {_fmt(r.get('defense_epochs'), '.0f', 6)} "
              f"{_fmt(r.get('attack_iterations'), ',.0f', 11)} {_fmt(r.get('defense_calls'), ',.0f', 9)} "
              f"{_fmt(r.get('accuracy_no_defense'), '.2f', 7)} {_fmt(r.get('accuracy_with_defense'), '.2f', 7)} "
              f"{_fmt(r.get('delta_accuracy'), '+.2f', 7)} {r.get('metric', ''):>8} "
              f"{_fmt(r.get('metric_no_defense'), spec, 8)} {_fmt(r.get('metric_with_defense'), spec, 8)}  "
              f"{r['verdict']:<12} {r['defense_working']:<20}")
    print("-" * 178)
    print("  AtkEp = attack epochs | DefEp = epochs the defense was active during training | Iterations = attack "
          "optimisation steps (unit in CSV) | DefCalls = batches the defense processed")
    print("  Acc0/Acc = main-task accuracy without/with defense | Before/After = attack metric without/with defense")
    print(f"  SUCCESS if LeakAUC < {THRESHOLDS['leak_auc_success']}, ASR < {THRESHOLDS['asr_success']}%, SSIM < "
          f"{THRESHOLDS['ssim_success']} | accuracy loss > {THRESHOLDS['utility_tolerance_pct']} pts flagged | "
          "ATTACK-WEAK = attack failed even without defense")
    print(line)


def print_defense_attack_matrix(rows):
    rows = [r for r in rows if r['verdict'] != BASELINE]
    if not rows:
        return
    attacks = list(dict.fromkeys(r['attack'] for r in rows))
    defenses = list(dict.fromkeys(r['defense'] for r in rows))
    cell = {(r['defense'], r['attack']): r['defense_working'] for r in rows}
    width = max(15, max(len(a) for a in attacks) + 2)
    print("\n" + "=" * (26 + width * len(attacks)))
    print("   WHICH DEFENSE SETTING WORKS AGAINST WHICH ATTACK")
    print("=" * (26 + width * len(attacks)))
    print(f"  {'Defense':<22}" + "".join(f"{a:>{width}}" for a in attacks))
    print("-" * (26 + width * len(attacks)))
    for d in defenses:
        print(f"  {d:<22.22}" + "".join(f"{cell.get((d, a), '.')[:width - 2]:>{width}}" for a in attacks))
    print("=" * (26 + width * len(attacks)))
    for d in defenses:
        mine = [r for r in rows if r['defense'] == d]
        works = [r['attack'] for r in mine if r['verdict'] == SUCCESS]
        partial = [r['attack'] for r in mine if r['verdict'] == PARTIAL]
        fails = [r['attack'] for r in mine if r['verdict'] == FAILED]
        print(f"  {d:<22} successful against: {', '.join(works) or '-'} | partial: {', '.join(partial) or '-'} | "
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


def call_with_supported_args(fn, *args, **kwargs):
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


def _metric(d, *names):
    for n in names:
        if n in d:
            return float(d[n])
    return float('nan')


def recon_budget(attack_key, n_train_batches):
    if attack_key == "whitebox":
        return 0, ITERATIONS * MAX_IMAGES, "inversion steps (steps per image x images)"
    if attack_key == "unsplit":
        n_batches = max(1, MAX_IMAGES // Config.BATCH_SIZE)
        main_iters = getattr(Config, "unsplit_main_iters", None)
        if main_iters is not None:
            inner = getattr(Config, "unsplit_input_iters", 0) + getattr(Config, "unsplit_model_iters", 0)
            return 0, main_iters * inner * n_batches, "optimisation steps (main x (input + model) iters x batches)"
        return 0, UNSPLIT_STEPS * n_batches, "inversion steps x batches"
    if attack_key == "ae_decoder":
        pairs = min(80, n_train_batches) * Config.BATCH_SIZE
        return AE_EPOCHS, AE_EPOCHS * math.ceil(0.9 * pairs / 32), "decoder training steps (AE epochs x AE batches)"
    return FSHA_EPOCHS, FSHA_EPOCHS * n_train_batches * 7, "optimisation steps (epochs x batches x (pilot + 5 critic + hijack))"


def summary_row(attack_key, defense_tag, model_label, cut_label, **fields):
    name, family = ATTACK_INFO[attack_key]
    row = {"dataset": Config.DATASET, "model": model_label, "cut_layer": cut_label, "attack": attack_key,
           "attack_name": name, "attack_family": family, "defense": defense_tag}
    row.update(fields)
    return row


def na_row(attack_key, defense_tag, model_label, cut_label, reason):
    return finalize_row(summary_row(attack_key, defense_tag, model_label, cut_label, verdict=NOT_APPLICABLE,
                                    reason=reason, defense_epochs=0, defense_calls=0))


def recon_rows(base, defended, defense_tag, model_label, cut_label, n_train_batches, defense_epochs, stage,
               acc0, acc1, na_reasons):
    rows = []
    for attack_key in ATTACKS_TO_RUN:
        if ATTACK_INFO[attack_key][1] != "reconstruction":
            continue
        if attack_key in na_reasons:
            rows.append(na_row(attack_key, defense_tag, model_label, cut_label, na_reasons[attack_key]))
            continue
        k = RECON_KEYS[attack_key]
        if f"{k}_ssim" not in defended:
            continue
        ep, it, unit = recon_budget(attack_key, n_train_batches)
        rows.append(finalize_row(summary_row(
            attack_key, defense_tag, model_label, cut_label,
            attack_epochs=ep, attack_iterations=it, iteration_unit=unit, defense_epochs=defense_epochs,
            defense_stage=stage, defense_calls=float('nan'),
            accuracy_no_defense=acc0, accuracy_with_defense=acc1,
            metric_no_defense=base.get(f"{k}_ssim", float('nan')) if base else float('nan'),
            metric_with_defense=defended[f"{k}_ssim"],
            psnr_no_defense=base.get(f"{k}_psnr", float('nan')) if base else float('nan'),
            psnr_with_defense=defended.get(f"{k}_psnr", float('nan')))))
    return rows


def print_run_info(r):
    print(f"\n  [run info] attack={r['attack']} | defense={r['defense']} | cut={r['cut_layer']} | "
          f"attack epochs={r.get('attack_epochs')} | defense epochs={r.get('defense_epochs')} | "
          f"iterations={r.get('attack_iterations')} | defense calls={r.get('defense_calls')}")
    print(f"  Result: {r['defense_working']} -- {r.get('reason', '')}")


from existing_defense.r3elu_defense import R3eLU


def model_tag():
    return {"KAGN": "kagn_sl", "PyramidCNN": "pyramidcnn_sl"}.get(Config.MODEL_NAME, "vanilla_sl")


def build_server_n(in_channels, num_classes):
    if Config.MODEL_NAME == "KAGN":
        return KAGNServerModel(cut_layer=Config.CUT_LAYER, num_classes=num_classes,
                               in_channels=in_channels, degree=Config.DEGREE)
    if Config.MODEL_NAME == "PyramidCNN":
        return PyramidCNNServerModel(cut_layer=Config.CUT_LAYER, num_classes=num_classes, in_channels=in_channels)
    return ServerModel(num_classes=num_classes)


def make_mech(eps, n_features, device):
    return R3eLUMechanism(n_features, epsilon=eps, K=int(K_FRACTION * n_features), C=C_CLIP,
                          dynamic_budget=DYNAMIC_BUDGET, protect_backward=PROTECT_BACKWARD, device=device)


class _R3eLUGradPerturb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mech):
        ctx.mech = mech
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        if ctx.mech.protect_backward:
            grad = ctx.mech.backward_perturb(grad)
        return grad, None


class R3eLUProtectedServer(nn.Module):
    def __init__(self, server_model, mech):
        super().__init__()
        self.server_model = server_model
        self.mech = mech
        self.r3elu = R3eLU(mech)
        self.calls = 0

    def forward(self, smashed):
        x = self.r3elu(smashed)
        if torch.is_grad_enabled() and smashed.requires_grad:
            x = _R3eLUGradPerturb.apply(x, self.mech)
            self.mech.end_iteration()
            self.calls += 1
        return self.server_model(x)


class R3eLUView:
    def __init__(self, mech):
        self.mech = mech
        self.calls = 0

    def protect(self, smashed_data, labels=None):
        self.calls += 1
        return self.mech.forward_perturb(smashed_data)


def _seed():
    torch.manual_seed(0); np.random.seed(0); random.seed(0)


def r3elu_label_leakage(eps, n_features, base_dataset, in_channels):
    from all_attacks.label_leakage_attack import GradientNormLabelLeakageAttack, build_binary_split_loaders
    _seed()
    loaders = build_binary_split_loaders(base_dataset, target_class=LEAK_POS_CLASS, positive_ratio=LEAK_POS_RATIO,
                                         batch_size=LEAK_BATCH, seed=0)
    attack = GradientNormLabelLeakageAttack(client_model=build_client(in_channels),
                                            server_model=build_server_n(in_channels, 1), dataset=Config.DATASET,
                                            target_class=LEAK_POS_CLASS, learning_rate=LEAK_LR)
    wrapper = None
    if eps is not None:
        wrapper = R3eLUProtectedServer(attack.server_model, make_mech(eps, n_features, attack.device))
        attack.server_model = wrapper
    summary = attack.run(loaders[0], loaders[1], epochs=LEAK_EPOCHS)
    attack.client_model.eval(); attack.server_model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loaders[1]:
            logits = attack.server_model(attack.client_model(x.to(attack.device))).squeeze(1)
            correct += ((logits.cpu() > 0).float() == y.float()).sum().item()
            total += len(y)
    return {"acc": 100.0 * correct / total, "attack_epochs": LEAK_EPOCHS,
            "metric": float(np.nanmax([summary["q95_norm_leak_auc_cut"], summary["q95_cosine_leak_auc_cut"]])),
            "iterations": len(loaders[0]) * LEAK_EPOCHS, "unit": "training steps (train batches x epochs)",
            "calls": wrapper.calls if wrapper else 0}


def r3elu_poison_client(eps, n_features, base_dataset, train_loader, test_loader, in_channels):
    from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack
    _seed()
    attack = BackdoorPoisonAttack(client_model=build_client(in_channels),
                                  server_model=build_server_n(in_channels, Config.NUM_CLASSES),
                                  base_dataset=base_dataset, dataset=Config.DATASET, num_classes=Config.NUM_CLASSES,
                                  mode="client", target_label=POISON_TARGET, poison_rate=POISON_RATE,
                                  patch_size=POISON_PATCH, trigger_value=POISON_TRIGGER, model_tag=model_tag())
    attack._save_checkpoint = lambda *a, **k: None
    attack.load_clean_init(vanilla_checkpoint_path())
    wrapper = None
    if eps is not None:
        wrapper = R3eLUProtectedServer(attack.server_model, make_mech(eps, n_features, attack.device))
        attack.server_model = wrapper
    attack.train(train_loader, test_loader, epochs=POISON_TRAIN_EPOCHS)
    cda, asr = attack.evaluate(test_loader)
    return {"acc": cda, "metric": asr, "attack_epochs": POISON_TRAIN_EPOCHS,
            "iterations": len(train_loader) * POISON_TRAIN_EPOCHS, "unit": "training steps (train batches x epochs)",
            "calls": wrapper.calls if wrapper else 0}


def r3elu_similarity(eps, n_features, base_dataset, in_channels):
    _seed()
    loader = DataLoader(base_dataset, batch_size=LEAK_BATCH, shuffle=True)
    attack = SimilarityLabelInferenceAttack(client_model=build_client(in_channels),
                                            server_model=build_server_n(in_channels, Config.NUM_CLASSES),
                                            dataset=Config.DATASET, num_classes=Config.NUM_CLASSES,
                                            distance='cosine', victim='gradient')
    wrapper = None
    if eps is not None:
        wrapper = R3eLUProtectedServer(attack.server_model, make_mech(eps, n_features, attack.device))
        attack.server_model = wrapper
    summary = attack.run(loader, epochs=SIM_EPOCHS)
    return {"acc": float('nan'), "metric": summary['cluster_accuracy'], "attack_epochs": SIM_EPOCHS,
            "iterations": len(loader) * SIM_EPOCHS, "unit": "training steps (batches x epochs)",
            "calls": wrapper.calls if wrapper else 0}


TRAINING_ATTACKS = {"label_leakage": r3elu_label_leakage, "similarity_label": r3elu_similarity,
                    "poison_client": r3elu_poison_client}



def run_pcat_defended(raw, device, n_features, model_label, cut_label, all_rows, all_out):
    def one(eps):
        hook = R3eLU(make_mech(eps, n_features, device)).to(device) if eps is not None else None
        atk = PCATAttack(Config.MODEL_NAME, Config.CUT_LAYER, Config.DATASET, raw, device,
                         defense_hook=hook, epochs=EPOCHS_PER_EPS, vanilla_iters=PCAT_VANILLA_ITERS,
                         tag='r3elu' if eps is not None else 'no_defense')
        return atk.run()
    base = None
    try:
        base = one(None)
    except Exception:
        traceback.print_exc()
    for eps in EPSILON_VALUES:
        try:
            r = one(eps)
            row = finalize_row(summary_row('pcat', f'r3elu_eps{eps}', model_label, cut_label,
                attack_epochs=r['epochs'], defense_epochs=r['epochs'],
                defense_stage='SL with R3eLU-forward on smashed data', attack_iterations=float('nan'),
                iteration_unit='SL epochs', defense_calls=float('nan'),
                accuracy_no_defense=base['victim_acc'] if base else float('nan'),
                accuracy_with_defense=r['victim_acc'],
                metric_no_defense=base['pcat_stolen_acc'] if base else float('nan'),
                metric_with_defense=r['pcat_stolen_acc'], note=f"independent {r['independent_acc']:.1f}%"))
        except Exception as exc:
            traceback.print_exc()
            row = na_row('pcat', f'r3elu_eps{eps}', model_label, cut_label, f"run failed: {type(exc).__name__}: {exc}")
        print_run_info(row); all_rows.append(row); save_results(all_rows, all_out)


def run_property_defended(raw, device, n_features, model_label, cut_label, all_rows, all_out):
    def best(eps):
        rows = []
        for prop in PROPERTIES[Config.DATASET]:
            atk = MaoPropertyInference(Config.MODEL_NAME, Config.CUT_LAYER, Config.DATASET, raw, device, prop,
                                       n_aux=PROP_N_AUX, n_eval=PROP_N_EVAL, attack_epochs=PROP_EPOCHS,
                                       victim_epochs=EPOCHS_PER_EPS, tag='r3elu' if eps is not None else 'no_defense')
            if eps is not None:
                mech = make_mech(eps, n_features, device)
                atk.client_forward = (lambda m, a: (lambda x: m.forward_perturb(a.client(x))))(mech, atk)
            rows.append(atk.run())
        return max(rows, key=lambda r: r['advantage_over_chance'])
    base = None
    try:
        base = best(None)
    except Exception:
        traceback.print_exc()
    for eps in EPSILON_VALUES:
        try:
            r = best(eps)
            row = finalize_row(summary_row('property', f'r3elu_eps{eps}', model_label, cut_label,
                attack_epochs=PROP_EPOCHS, defense_epochs=EPOCHS_PER_EPS,
                defense_stage='inference: R3eLU-forward on smashed data', attack_iterations=float('nan'),
                iteration_unit='F training epochs', defense_calls=float('nan'),
                accuracy_no_defense=base['victim_test_acc'] if base else float('nan'),
                accuracy_with_defense=r['victim_test_acc'],
                metric_no_defense=base['pia_accuracy'] if base else float('nan'),
                metric_with_defense=r['pia_accuracy'], note=f"best property {r['property']}"))
        except Exception as exc:
            traceback.print_exc()
            row = na_row('property', f'r3elu_eps{eps}', model_label, cut_label, f"run failed: {type(exc).__name__}: {exc}")
        print_run_info(row); all_rows.append(row); save_results(all_rows, all_out)


def run_membership_defended(raw, device, n_features, model_label, cut_label, all_rows, all_out):
    def predict_defended(mech):
        def pf(client, server, x):
            return torch.softmax(server(mech.forward_perturb(client(x))), dim=1)
        return pf
    base = None
    try:
        base = ShokriMembershipInference(Config.MODEL_NAME, Config.CUT_LAYER, Config.DATASET, raw, device,
                                         target_size=MIA_TARGET, num_shadows=MIA_SHADOWS, epochs=EPOCHS_PER_EPS,
                                         tag='no_defense').run()
    except Exception:
        traceback.print_exc()
    for eps in EPSILON_VALUES:
        try:
            mech = make_mech(eps, n_features, device)
            r = ShokriMembershipInference(Config.MODEL_NAME, Config.CUT_LAYER, Config.DATASET, raw, device,
                                          target_size=MIA_TARGET, num_shadows=MIA_SHADOWS, epochs=EPOCHS_PER_EPS,
                                          predict_fn=predict_defended(mech), tag=f'r3elu_eps{eps}').run()
            row = finalize_row(summary_row('membership', f'r3elu_eps{eps}', model_label, cut_label,
                attack_epochs=EPOCHS_PER_EPS, defense_epochs=0,
                defense_stage='inference: R3eLU-forward on the prediction vector', attack_iterations=float('nan'),
                iteration_unit='shadow epochs', defense_calls=float('nan'),
                accuracy_no_defense=base['target_train_acc'] if base else float('nan'),
                accuracy_with_defense=r['target_train_acc'],
                metric_no_defense=base['mia_auc'] if base else float('nan'),
                metric_with_defense=r['mia_auc'], note=f"attack acc {r['mia_accuracy']:.1f}%"))
        except Exception as exc:
            traceback.print_exc()
            row = na_row('membership', f'r3elu_eps{eps}', model_label, cut_label, f"run failed: {type(exc).__name__}: {exc}")
        print_run_info(row); all_rows.append(row); save_results(all_rows, all_out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=Config.MODEL_NAME, choices=['KAGN', 'PyramidCNN'])
    ap.add_argument('--cut', type=int, default=Config.CUT_LAYER, choices=[1, 2, 3, 4])
    ap.add_argument('--dataset', default=Config.DATASET, choices=['CIFAR10', 'MNIST'])
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--all-eps', action='store_true')
    _a = ap.parse_args()
    Config.MODEL_NAME, Config.CUT_LAYER, Config.DATASET = _a.model, _a.cut, _a.dataset
    _VALID = DE.load_valid_configs()
    if not _a.all_eps:
        EPSILON_VALUES = [1.0]
    if _a.quick:
        EPSILON_VALUES, EPOCHS_PER_EPS = [1.0], 1
        MAX_IMAGES, ITERATIONS, AE_EPOCHS, FSHA_EPOCHS = 4, 20, 2, 1
        LEAK_EPOCHS = SIM_EPOCHS = POISON_TRAIN_EPOCHS = 1
        FORA_ITERS = 30
        PCAT_VANILLA_ITERS, PROP_N_AUX, PROP_N_EVAL, PROP_EPOCHS = 50, 500, 1000, 3
        MIA_SHADOWS, MIA_TARGET = 2, 300
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")
    train_loader, test_loader = DatasetLoader(dataset_name=Config.DATASET).get_loaders()
    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    ckpt_path = vanilla_checkpoint_path()
    vanilla = torch.load(ckpt_path, map_location=device) if os.path.exists(ckpt_path) else None
    print(f"[{'✓' if vanilla else '!'}] Vanilla checkpoint: {ckpt_path if vanilla else 'not found — training from scratch'}")

    client, server = build_client(in_channels).to(device), build_server(in_channels).to(device)
    if vanilla:
        client.load_state_dict(vanilla['client_state']); server.load_state_dict(vanilla['server_state'])
    with torch.no_grad():
        n_features = client(next(iter(test_loader))[0][:1].to(device)).numel()
    base_mech = R3eLUMechanism(n_features, epsilon=1.0, protect_forward=False, protect_backward=False, device=device)
    base_trainer = R3eLUSplitLearningTrainer(client, server, train_loader, test_loader, base_mech, tag="r3elu_baseline")
    if not vanilla:
        Config.EPOCHS = EPOCHS_PER_EPS
        base_trainer.train()
    base_acc = base_trainer._evaluate()
    print("\n" + "=" * 60 + "\n  BASELINE (NO DEFENSE)\n" + "=" * 60 + f"\n  Accuracy: {base_acc:.2f}%")
    results = [{'epsilon': 'inf (No Defense)', 'accuracy': base_acc,
                **attack_suite(client, lambda z: z, train_loader, test_loader, device, in_channels, 'no_defense_r3elu')}]

    for eps in EPSILON_VALUES:
        print("\n" + "=" * 60 + f"\n  R3eLU DEFENSE — epsilon = {eps}\n" + "=" * 60)
        client, server = build_client(in_channels).to(device), build_server(in_channels).to(device)
        if vanilla:
            client.load_state_dict(vanilla['client_state']); server.load_state_dict(vanilla['server_state'])
        mech = R3eLUMechanism(n_features, epsilon=eps, K=int(K_FRACTION * n_features), C=C_CLIP,
                              dynamic_budget=DYNAMIC_BUDGET, protect_backward=PROTECT_BACKWARD, device=device)
        print(f"  {mech}")
        Config.EPOCHS = EPOCHS_PER_EPS
        trainer = R3eLUSplitLearningTrainer(client, server, train_loader, test_loader, mech, tag=f"r3elu_eps{eps}")
        trainer.train()
        trainer.save_results()
        acc = trainer._evaluate()
        sampling = Config.BATCH_SIZE / len(train_loader.dataset)
        eps_total = mech.accountant(total_steps=mech.step, sampling_ratio=sampling)
        print(f"  Accuracy WITH defense: {acc:.2f}% (drop {base_acc - acc:.2f}) | "
              f"strong-composition eps_total over {mech.step} steps: {eps_total:.2f}")
        results.append({'epsilon': eps, 'accuracy': acc, 'eps_total_strong_comp': eps_total,
                        **attack_suite(client, mech.forward_perturb, train_loader, test_loader, device,
                                       in_channels, f'r3elu_eps{eps}')})

    df = pd.DataFrame(results)
    out = f"{Config.RESULTS_DIR}/r3elu_defense_evaluation_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv"
    df.to_csv(out, index=False)

    cols = [c for c in ['wb_psnr', 'wb_ssim', 'us_psnr', 'us_ssim', 'ae_psnr', 'ae_ssim', 'fsha_psnr', 'fsha_ssim'] if c in df]
    print("\n" + "=" * 100)
    print(f" THESIS TABLE — ATTACKS VS R3eLU (Mao et al.) — {Config.MODEL_NAME} / {Config.DATASET}")
    print("=" * 100)
    print(f"{'Epsilon':<18}{'Acc (%)':<10}" + "".join(f"{c:<12}" for c in cols))
    print("-" * 100)
    for r in results:
        print(f"{str(r['epsilon']):<18}{r['accuracy']:<10.2f}" + "".join(f"{r.get(c, float('nan')):<12.4f}" for c in cols))
    print("=" * 100)
    print(f"\nSaved raw data → {out}")

    model_label = Config.MODEL_NAME
    cut_label = (Config.CUT_LAYER if Config.MODEL_NAME in ("KAGN", "PyramidCNN") else "fixed")
    base_dataset = train_loader.dataset
    n_train_batches = len(train_loader)
    stage = ("training + inference: R3eLU-forward on smashed data"
             + (", R3eLU-backward on gradients" if PROTECT_BACKWARD else ""))
    all_rows = []
    all_out = f"{Config.RESULTS_DIR}/r3elu_all_attacks_{Config.DATASET}.csv"

    print("\n" + "#" * 78)
    print("#   R3eLU vs ALL ATTACKS")
    print(f"#   MODEL: {model_label} | CUT LAYER: {cut_label} | {Config.DATASET}")
    print("#" * 78)

    for r in results[1:]:
        all_rows += recon_rows(results[0], r, f"r3elu_eps{r['epsilon']}", model_label, cut_label, n_train_batches,
                               EPOCHS_PER_EPS, stage, base_acc, r['accuracy'], {})

    for attack_key, fn in TRAINING_ATTACKS.items():
        if attack_key not in ATTACKS_TO_RUN or not attack_allowed(attack_key):
            continue
        print("\n" + "#" * 78)
        print(f"#   R3eLU vs {attack_key} -- {ATTACK_INFO[attack_key][0]}")
        print(f"#   MODEL: {model_label} | CUT LAYER: {cut_label} | {Config.DATASET}")
        print("#" * 78)
        args = {"label_leakage": (n_features, base_dataset, in_channels),
                "similarity_label": (n_features, base_dataset, in_channels),
                "poison_client": (n_features, base_dataset, train_loader, test_loader, in_channels)}[attack_key]
        try:
            print(f"\n  >> {attack_key}: baseline (no defense)")
            base = fn(None, *args)
        except Exception:
            traceback.print_exc()
            base = None
        for eps in EPSILON_VALUES:
            tag = f"r3elu_eps{eps}"
            try:
                print(f"\n  >> {attack_key}: R3eLU eps = {eps}")
                res = fn(eps, *args)
                row = summary_row(attack_key, tag, model_label, cut_label,
                                  attack_epochs=res["attack_epochs"], defense_epochs=res["attack_epochs"],
                                  defense_stage=stage, attack_iterations=res["iterations"],
                                  iteration_unit=res["unit"], defense_calls=res["calls"],
                                  accuracy_no_defense=base["acc"] if base else float('nan'),
                                  accuracy_with_defense=res["acc"],
                                  metric_no_defense=base["metric"] if base else float('nan'),
                                  metric_with_defense=res["metric"])
                if "lia" in res and base:
                    row["note"] = f"label inference {base['lia']:.1f}% -> {res['lia']:.1f}%"
                finalize_row(row)
            except Exception as exc:
                traceback.print_exc()
                row = na_row(attack_key, tag, model_label, cut_label, f"run failed: {type(exc).__name__}: {exc}")
            print_run_info(row)
            all_rows.append(row)
            save_results(all_rows, all_out)

    if "poison_server" in ATTACKS_TO_RUN and attack_allowed("poison_server"):
        print("\n" + "#" * 78)
        print(f"#   R3eLU vs poison_server -- {ATTACK_INFO['poison_server'][0]}")
        print("#" * 78)
        from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack
        atk = BackdoorPoisonAttack(client_model=build_client(in_channels),
                                   server_model=build_server_n(in_channels, Config.NUM_CLASSES),
                                   base_dataset=base_dataset, dataset=Config.DATASET, num_classes=Config.NUM_CLASSES,
                                   mode="server", target_label=POISON_TARGET, poison_rate=POISON_RATE,
                                   patch_size=POISON_PATCH, trigger_value=POISON_TRIGGER,
                                   surrogate_builder=lambda: build_client(in_channels), model_tag=model_tag())
        print(f"  Poisoned checkpoint path: {atk._checkpoint_path()}")
        if not atk.load_checkpoint():
            print(f"\n[!] No poisoned checkpoint found at: {atk._checkpoint_path()}")
            print("    Run attack_runner/run_backdoor_poison.py first to produce it. Skipping.")
            for eps in EPSILON_VALUES:
                all_rows.append(na_row("poison_server", f"r3elu_eps{eps}", model_label, cut_label,
                                       f"no poisoned checkpoint at {atk._checkpoint_path()}"))
        else:
            ep = len(atk.history["epoch"]) if atk.history.get("epoch") else float('nan')
            cda0, asr0 = atk.evaluate(test_loader)
            for eps in EPSILON_VALUES:
                view = R3eLUView(make_mech(eps, n_features, device))
                cda, asr = atk.evaluate(test_loader, defense=view)
                row = finalize_row(summary_row(
                    "poison_server", f"r3elu_eps{eps}", model_label, cut_label,
                    attack_epochs=ep, defense_epochs=0, defense_stage="inference: R3eLU-forward on smashed data",
                    attack_iterations=ep * n_train_batches if ep == ep else float('nan'),
                    iteration_unit="training steps (epochs up to best-ASR checkpoint x train batches)",
                    defense_calls=view.calls, accuracy_no_defense=cda0, accuracy_with_defense=cda,
                    metric_no_defense=asr0, metric_with_defense=asr,
                    note="the server is the attacker, so only the guest-side R3eLU-forward applies; the poisoned "
                         "server was not trained with R3eLU noise"))
                print_run_info(row)
                all_rows.append(row)
        save_results(all_rows, all_out)

    if any(attack_allowed(k) for k in ('pcat', 'property', 'membership')):
        raw_env = load_raw_datasets(Config.DATASET)
        if attack_allowed('pcat'):
            run_pcat_defended(raw_env, device, n_features, model_label, cut_label, all_rows, all_out)
        if attack_allowed('property'):
            run_property_defended(raw_env, device, n_features, model_label, cut_label, all_rows, all_out)
        if attack_allowed('membership'):
            run_membership_defended(raw_env, device, n_features, model_label, cut_label, all_rows, all_out)

    order = {a: i for i, a in enumerate(ATTACKS_TO_RUN)}
    all_rows.sort(key=lambda r: order.get(r["attack"], 99))
    for r in all_rows:
        if r["attack_family"] == "reconstruction":
            print_run_info(r)
    print_summary_table(all_rows, f"R3eLU vs ALL ATTACKS -- {model_label} | cut {cut_label} | {Config.DATASET}")
    print_defense_attack_matrix(all_rows)
    save_results(all_rows, all_out)
    print(f"\n  Saved -> {all_out}")