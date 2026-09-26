import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect
import math
import random
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from dataset import DatasetLoader
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel
from all_defences.protoguard_sl_defense import ProtoGuardSLDefense


SEED = 0

ALPHA              = 0.5
CALIBRATION_EPOCHS = 1
CALIB_BATCHES      = 100
REFIT_EVERY        = 0
TRAINING_STAGE     = "both"

MODES_TO_RUN = ["client", "server"]

ATTACKS_TO_RUN = ["label_leakage", "villain", "poison_client", "poison_server",
                  "whitebox", "unsplit", "ae_decoder", "fsha"]

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

TARGET_LABEL  = 0
POISON_RATE   = 0.05
PATCH_SIZE    = 4
TRIGGER_VALUE = 1.0
ATTACK_EPOCHS_OVERRIDE = None

CLEAN_EPOCHS      = 5
RECON_IMAGES      = 32
WHITEBOX_ITERS    = 1000
UNSPLIT_STEPS     = 1000
AE_EPOCHS         = 50
FSHA_EPOCHS       = 5
FSHA_CRITIC_ITERS = 5

QUICK_TEST = False

ATTACKS = {
    "label_leakage": ("Label Leakage (norm + cosine gradient scoring)", "label_inference", "client"),
    "villain":       ("VILLAIN backdoor (label inference + embedding trigger)", "backdoor", "client"),
    "poison_client": ("Backdoor poisoning, client attacker (BadNets patch)", "backdoor", "client"),
    "poison_server": ("Backdoor poisoning, server attacker (surrogate client)", "backdoor", "server"),
    "whitebox":      ("White-box model inversion", "reconstruction", "server"),
    "unsplit":       ("UnSplit model inversion", "reconstruction", "server"),
    "ae_decoder":    ("AE decoder inversion", "reconstruction", "server"),
    "fsha":          ("FSHA feature-space hijacking", "reconstruction", "server"),
}


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
    'replaced_pct', 'accuracy_no_defense', 'accuracy_with_defense', 'delta_accuracy',
    'metric', 'metric_no_defense', 'metric_with_defense',
    'verdict', 'defense_working', 'threat_model', 'reason',
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
    psnr_drop = psnr_base - psnr_dfd if not (_missing(psnr_base) or _missing(psnr_dfd)) else 0.0
    if base - dfd >= drop or (family == "reconstruction" and psnr_drop >= t['psnr_partial_drop']):
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
    verdict, reason = judge(row['attack_family'], row.get('metric_no_defense'), row.get('metric_with_defense'),
                            row.get('psnr_no_defense'), row.get('psnr_with_defense'))
    utility_ok = True
    if not _missing(row['delta_accuracy']):
        utility_ok = row['delta_accuracy'] >= -THRESHOLDS['utility_tolerance_pct']
        reason += f"; accuracy {row['delta_accuracy']:+.2f} pts"
    if not _missing(row.get('replaced_pct')):
        reason += f"; {row['replaced_pct']:.1f}% samples replaced by prototype"
    if str(row.get('threat_model', '')).startswith("mismatch"):
        reason += " [threat-model mismatch]"
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
    line = "=" * 184
    print("\n" + line)
    print(f"   SUMMARY -- {title}")
    print(line)
    print(f"  {'Attack':<14} {'Defense':<16} {'Cut':>5} {'AtkEp':>6} {'DefEp':>6} {'Iterations':>11} "
          f"{'DefCalls':>9} {'Repl%':>6} {'Acc0%':>7} {'Acc%':>7} {'dAcc':>7} {'Metric':>8} {'Before':>8} "
          f"{'After':>8}  {'Verdict':<12} {'Working':<20}")
    print("-" * 184)
    for r in rows:
        spec = '.2f' if r['attack_family'] == 'backdoor' else '.4f'
        print(f"  {r['attack']:<14} {r['defense']:<16.16} {str(r['cut_layer']):>5} "
              f"{_fmt(r.get('attack_epochs'), '.0f', 6)} {_fmt(r.get('defense_epochs'), '.0f', 6)} "
              f"{_fmt(r.get('attack_iterations'), ',.0f', 11)} {_fmt(r.get('defense_calls'), ',.0f', 9)} "
              f"{_fmt(r.get('replaced_pct'), '.1f', 6)} "
              f"{_fmt(r.get('accuracy_no_defense'), '.2f', 7)} {_fmt(r.get('accuracy_with_defense'), '.2f', 7)} "
              f"{_fmt(r.get('delta_accuracy'), '+.2f', 7)} {r.get('metric', ''):>8} "
              f"{_fmt(r.get('metric_no_defense'), spec, 8)} {_fmt(r.get('metric_with_defense'), spec, 8)}  "
              f"{r['verdict']:<12} {r['defense_working']:<20}")
    print("-" * 184)
    print("  AtkEp = attack training epochs | DefEp = epochs the defense was active or fitted | Iterations = attack "
          "optimisation steps (unit in CSV) | DefCalls = batches ProtoGuard processed")
    print("  Repl% = samples replaced by a class prototype | Acc0/Acc = main-task accuracy without/with defense | "
          "Before/After = attack metric without/with defense")
    print(f"  SUCCESS if LeakAUC < {THRESHOLDS['leak_auc_success']}, ASR < {THRESHOLDS['asr_success']}%, SSIM < "
          f"{THRESHOLDS['ssim_success']} | accuracy loss > {THRESHOLDS['utility_tolerance_pct']} pts flagged | "
          "ATTACK-WEAK = attack failed even without defense")
    print("  Threat model per attack is in the CSV (column threat_model).")
    print(line)
    judged = [r for r in rows if r['verdict'] not in (BASELINE,)]
    works = [r['attack'] for r in judged if r['verdict'] == SUCCESS]
    partial = [r['attack'] for r in judged if r['verdict'] == PARTIAL]
    fails = [r['attack'] for r in judged if r['verdict'] == FAILED]
    weak = [r['attack'] for r in judged if r['verdict'] == WEAK]
    na = [r['attack'] for r in judged if r['verdict'] == NOT_APPLICABLE]
    print(f"\n  ProtoGuard-SL (alpha={ALPHA}) is successful against: {', '.join(works) or '-'}")
    print(f"                                partial against     : {', '.join(partial) or '-'}")
    print(f"                                fails against       : {', '.join(fails) or '-'}")
    if weak:
        print(f"                                attack too weak     : {', '.join(weak)}")
    if na:
        print(f"                                not run / failed    : {', '.join(na)}")


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


CLEAN_CHECKPOINT_TAGS = {"Vanilla": "vanilla_sl", "PyramidCNN": "pyramidcnn_sl", "KAGN": "kagn_sl"}


def build_split_models(model_name, in_channels, cut_layer, num_classes):
    if model_name == "KAGN":
        return (KAGNClientModel(cut_layer=cut_layer, in_channels=in_channels, degree=Config.DEGREE),
                KAGNServerModel(cut_layer=cut_layer, num_classes=num_classes, in_channels=in_channels,
                                degree=Config.DEGREE))
    if model_name == "PyramidCNN":
        return (PyramidCNNClientModel(cut_layer=cut_layer, in_channels=in_channels),
                PyramidCNNServerModel(cut_layer=cut_layer, num_classes=num_classes, in_channels=in_channels))
    return ClientModel(in_channels=in_channels), ServerModel(num_classes=num_classes)


def denormalize(inputs):
    if Config.DATASET == "CIFAR10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
    else:
        mean, std = torch.tensor([0.1307]).view(1, 1, 1, 1), torch.tensor([0.3081]).view(1, 1, 1, 1)
    return torch.clamp(inputs * std.to(inputs.device) + mean.to(inputs.device), 0, 1)


def call_with_supported_args(fn, *args, **kwargs):
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


def metric(d, *names):
    for n in names:
        if n in d:
            return float(d[n])
    return float('nan')


class LabelHolder:
    def __init__(self):
        self.current = None

    def labels_for(self, n):
        if self.current is None or len(self.current) < n:
            return None
        return self.current[:n]


class TappedLoader:
    def __init__(self, loader, holder, on_epoch_start=None, refit_every=0):
        self.loader, self.holder = loader, holder
        self.on_epoch_start, self.refit_every = on_epoch_start, refit_every

    def __iter__(self):
        if self.on_epoch_start is not None:
            self.on_epoch_start()
        for i, batch in enumerate(self.loader):
            if self.on_epoch_start is not None and self.refit_every and i > 0 and i % self.refit_every == 0:
                self.on_epoch_start()
            labels = batch[1]
            self.holder.current = labels if torch.is_tensor(labels) else torch.as_tensor(labels)
            yield batch

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class LimitedLoader:
    def __init__(self, loader, max_batches):
        self.loader, self.max_batches = loader, max_batches

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            yield batch[0], batch[1]

    def __len__(self):
        return len(self.loader) if self.max_batches is None else min(self.max_batches, len(self.loader))


class ProtoGuardRuntime:

    def __init__(self, num_classes):
        self.impl = ProtoGuardSLDefense(num_classes=num_classes, alpha=ALPHA)
        self.fit_passes = self.calls = self.samples = self.replaced = 0

    def fit(self, client_model, calib_loader, device):
        was_training = client_model.training
        for _ in range(CALIBRATION_EPOCHS):
            self.impl.fit(client_model, calib_loader, device)
        client_model.train(was_training)
        self.fit_passes += 1

    def protect(self, smashed_data, labels=None):
        if labels is None or len(labels) < smashed_data.shape[0]:
            return smashed_data
        labels = labels[:smashed_data.shape[0]].to(smashed_data.device)
        out = self.impl.protect(smashed_data, labels)
        self.calls += 1
        self.samples += int(smashed_data.shape[0])
        with torch.no_grad():
            self.replaced += int((out != smashed_data).flatten(1).any(dim=1).sum().item())
        return out

    @property
    def replaced_pct(self):
        return 100.0 * self.replaced / self.samples if self.samples else float('nan')


class ProtectedServer(nn.Module):

    def __init__(self, server_model, runtime, holder, train_active=True, eval_active=True):
        super().__init__()
        self.server_model, self.runtime, self.holder = server_model, runtime, holder
        self.train_active, self.eval_active = train_active, eval_active

    def forward(self, smashed):
        active = self.train_active if torch.is_grad_enabled() else self.eval_active
        if active:
            labels = self.holder.labels_for(smashed.shape[0])
            if labels is not None and len(labels) == smashed.shape[0]:
                smashed = self.runtime.protect(smashed, labels)
        return self.server_model(smashed)


class ProtectedClient(nn.Module):

    def __init__(self, client_model, runtime, holder):
        super().__init__()
        self.client_model, self.runtime, self.holder = client_model, runtime, holder

    def forward(self, x):
        smashed = self.client_model(x)
        return self.runtime.protect(smashed, self.holder.labels_for(smashed.shape[0]))


@torch.no_grad()
def test_accuracy(client, server, loader, device, runtime=None):
    client.eval(); server.eval()
    correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        smashed = client(inputs)
        if runtime is not None:
            smashed = runtime.protect(smashed, labels)
        correct += server(smashed).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / max(total, 1)


def threat_model(attack_key):
    if ATTACKS[attack_key][2] == "client":
        return "valid (server-side defense vs client-side attacker)"
    return "mismatch (server is the attacker; shows the effect if the filter ran before it sees the data)"


class Experiment:
    def __init__(self, device):
        self.device = device
        self.model_name = model_from_config()
        self.cut = Config.CUT_LAYER
        self.cut_label = "fixed" if self.model_name == "Vanilla" else self.cut
        self.in_channels = 1 if Config.DATASET == "MNIST" else 3
        data = DatasetLoader(dataset_name=Config.DATASET)
        self.train_loader, self.test_loader = data.get_loaders()
        self.base_dataset = self.train_loader.dataset
        self.calib_loader = LimitedLoader(self.train_loader, CALIB_BATCHES)
        self._clean = None

    def row(self, attack_key, defense):
        name, family, _ = ATTACKS[attack_key]
        return {"dataset": Config.DATASET, "model": self.model_name, "cut_layer": self.cut_label,
                "attack": attack_key, "attack_name": name, "attack_family": family, "defense": defense}

    def pair(self, attack_key, shared, base, dfd, runtime, stage, defense_epochs, calls=None, replaced_pct=None):
        b = self.row(attack_key, "no_defense")
        b.update(shared)
        b.update({"defense_epochs": 0, "defense_calls": 0, "defense_stage": "none", "verdict": BASELINE,
                  "accuracy_no_defense": base["acc"], "accuracy_with_defense": base["acc"],
                  "metric_no_defense": base["metric"], "metric_with_defense": base["metric"]})
        d = self.row(attack_key, f"protoguard_a{ALPHA:g}")
        d.update(shared)
        d.update({"defense_epochs": defense_epochs,
                  "defense_calls": runtime.calls if calls is None else calls,
                  "replaced_pct": runtime.replaced_pct if replaced_pct is None else replaced_pct,
                  "defense_stage": stage,
                  "threat_model": threat_model(attack_key),
                  "accuracy_no_defense": base["acc"], "accuracy_with_defense": dfd["acc"],
                  "metric_no_defense": base["metric"], "metric_with_defense": dfd["metric"]})
        for k in ("psnr", "mse", "lia"):
            if k in base:
                d[f"{k}_no_defense"], d[f"{k}_with_defense"] = base[k], dfd[k]
        finalize_row(b)
        finalize_row(d)
        if "lia" in base:
            d["reason"] += f"; label inference {base['lia']:.1f}% -> {dfd['lia']:.1f}%"
        return [b, d]

    def _leak_train(self, loaders, runtime=None):
        from all_attacks.label_leakage_attack import GradientNormLabelLeakageAttack
        set_seed()
        train_loader, test_loader = loaders
        client, server = build_split_models(self.model_name, self.in_channels, self.cut, 1)
        attack = GradientNormLabelLeakageAttack(client_model=client, server_model=server, dataset=Config.DATASET,
                                                target_class=LEAK_POS_CLASS, learning_rate=LEAK_LR)
        holder = LabelHolder()
        if runtime is not None:
            calib = LimitedLoader(train_loader, CALIB_BATCHES)
            refit = lambda: runtime.fit(attack.client_model, calib, attack.device)
            attack.server_model = ProtectedServer(attack.server_model, runtime, holder,
                                                  train_active=TRAINING_STAGE in ("training", "both"),
                                                  eval_active=TRAINING_STAGE in ("inference", "both"))
            train_loader = TappedLoader(train_loader, holder, on_epoch_start=refit, refit_every=REFIT_EVERY)
        test_loader = TappedLoader(test_loader, holder)
        summary = attack.run(train_loader, test_loader, epochs=LEAK_EPOCHS)

        attack.client_model.eval(); attack.server_model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                logits = attack.server_model(attack.client_model(inputs.to(attack.device))).squeeze(1)
                correct += ((logits.cpu() > 0).float() == labels.float()).sum().item()
                total += len(labels)
        return {"acc": 100.0 * correct / total,
                "metric": float(np.nanmax([summary["q95_norm_leak_auc_cut"], summary["q95_cosine_leak_auc_cut"]]))}

    def run_label_leakage(self):
        from all_attacks.label_leakage_attack import build_binary_split_loaders
        loaders = build_binary_split_loaders(self.base_dataset, target_class=LEAK_POS_CLASS,
                                             positive_ratio=LEAK_POS_RATIO, batch_size=LEAK_BATCH, seed=SEED)
        print("\n  >> label_leakage: baseline (no defense)")
        base = self._leak_train(loaders)
        print(f"\n  >> label_leakage: ProtoGuard-SL (stage = {TRAINING_STAGE})")
        runtime = ProtoGuardRuntime(num_classes=2)
        dfd = self._leak_train(loaders, runtime)
        shared = {"attack_epochs": LEAK_EPOCHS, "attack_iterations": len(loaders[0]) * LEAK_EPOCHS,
                  "iteration_unit": "training steps (train batches x epochs)"}
        stage = f"{TRAINING_STAGE}: filters smashed data at the server (fitted {runtime.fit_passes}x)"
        on_in_training = TRAINING_STAGE in ("training", "both")
        return self.pair("label_leakage", shared, base, dfd, runtime, stage, LEAK_EPOCHS if on_in_training else 0)

    def _villain_train(self, runtime=None):
        from all_attacks.villain_backdoor_attack import VILLAINBackdoorAttack, build_indexed_loader
        set_seed()
        client, server = build_split_models(self.model_name, self.in_channels, self.cut, Config.NUM_CLASSES)
        attack = VILLAINBackdoorAttack(client_model=client, server_model=server, base_dataset=self.base_dataset,
                                       dataset=Config.DATASET, num_classes=Config.NUM_CLASSES,
                                       target_label=VILLAIN_TARGET, poison_rate=VILLAIN_POISON,
                                       candidates_per_batch=VILLAIN_CANDIDATES)
        inner_server = attack.server_model
        loader = build_indexed_loader(self.base_dataset, batch_size=VILLAIN_BATCH, shuffle=True)
        train_on = runtime is not None and TRAINING_STAGE in ("training", "both")
        state = {"active": False}
        if runtime is not None:
            holder = LabelHolder()
            refit = lambda: (runtime.fit(attack.client_model, self.calib_loader, self.device)
                             if state["active"] else None)
            attack.server_model = ProtectedServer(inner_server, runtime, holder, train_active=False,
                                                  eval_active=False)
            loader = TappedLoader(loader, holder, on_epoch_start=refit, refit_every=REFIT_EVERY)

        attack.warmup(loader, epochs=VILLAIN_WARMUP)
        if train_on:
            state["active"] = True
            attack.server_model.train_active = True
        attack.infer_labels(loader, epochs=VILLAIN_INFERENCE)
        attack.fabricate_trigger(loader)
        attack.inject_backdoor(loader, self.test_loader, epochs=VILLAIN_INJECTION)

        use_at_test = runtime is not None and TRAINING_STAGE in ("inference", "both")
        if use_at_test:
            runtime.fit(attack.client_model, self.calib_loader, self.device)
        attack.client_model.eval(); inner_server.eval()
        ok = n = hit = m = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                emb = attack.client_model(inputs)
                view = runtime.protect(emb, labels) if use_at_test else emb
                ok += inner_server(view).argmax(1).eq(labels).sum().item()
                n += labels.size(0)
                rows = labels != attack.target_label
                if attack.trigger is None or rows.sum() == 0:
                    continue
                trig = (emb.flatten(1)[rows] + attack.trigger).view(-1, *emb.shape[1:])
                if use_at_test:
                    trig = runtime.protect(trig, labels[rows])
                hit += inner_server(trig).argmax(1).eq(attack.target_label).sum().item()
                m += int(rows.sum())
        return {"acc": 100.0 * ok / n, "metric": 100.0 * hit / m if m else 0.0,
                "lia": 100.0 * attack.label_inference_accuracy(), "batches": len(loader)}

    def run_villain(self):
        print("\n  >> villain: baseline (no defense)")
        base = self._villain_train()
        print(f"\n  >> villain: ProtoGuard-SL (stage = {TRAINING_STAGE})")
        runtime = ProtoGuardRuntime(num_classes=Config.NUM_CLASSES)
        dfd = self._villain_train(runtime)
        total = VILLAIN_WARMUP + VILLAIN_INFERENCE + VILLAIN_INJECTION
        shared = {"attack_epochs": total,
                  "attack_epochs_detail": f"warmup {VILLAIN_WARMUP} + inference {VILLAIN_INFERENCE} + "
                                          f"injection {VILLAIN_INJECTION}",
                  "attack_iterations": base["batches"] * total,
                  "iteration_unit": "training steps (train batches x all VILLAIN epochs)"}
        stage = f"{TRAINING_STAGE}: filters smashed data at the server after warm-up (fitted {runtime.fit_passes}x)"
        on_in_training = TRAINING_STAGE in ("training", "both")
        return self.pair("villain", shared, base, dfd, runtime, stage,
                         VILLAIN_INFERENCE + VILLAIN_INJECTION if on_in_training else 0)

    def run_one(self, mode):
        from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack, DummyNoDefense
        attack_key = f"poison_{mode}"
        client, server = build_split_models(self.model_name, self.in_channels, self.cut, Config.NUM_CLASSES)
        surrogate = (lambda: build_split_models(self.model_name, self.in_channels, self.cut,
                                                Config.NUM_CLASSES)[0]) if mode == "server" else None
        attack = BackdoorPoisonAttack(client_model=client, server_model=server, base_dataset=self.base_dataset,
                                      dataset=Config.DATASET, num_classes=Config.NUM_CLASSES, mode=mode,
                                      target_label=TARGET_LABEL, poison_rate=POISON_RATE, patch_size=PATCH_SIZE,
                                      trigger_value=TRIGGER_VALUE, surrogate_builder=surrogate,
                                      model_tag=CLEAN_CHECKPOINT_TAGS[self.model_name])
        print("\n" + "#" * 70)
        print(f"#   PROTOGUARD-SL vs BACKDOOR POISONING ({mode.upper()}) "
              f"-- MODEL: {self.model_name}  |  CUT: {Config.CUT_LAYER}  |  DATASET: {Config.DATASET}")
        print("#" * 70)
        print(f"  Poisoned checkpoint path: {attack._checkpoint_path()}")
        if not attack.load_checkpoint():
            print(f"\n[!] No poisoned checkpoint found at: {attack._checkpoint_path()}")
            print("    Run defence_runner/run_backdoor_poison.py first to produce it. Skipping.")
            r = self.row(attack_key, f"protoguard_a{ALPHA:g}")
            r.update({"verdict": NOT_APPLICABLE, "threat_model": threat_model(attack_key),
                      "reason": f"no poisoned checkpoint at {attack._checkpoint_path()} "
                                f"(run run_backdoor_poison.py first with the same model / cut)"})
            return [finalize_row(r)]

        if ATTACK_EPOCHS_OVERRIDE is not None:
            epochs, source = ATTACK_EPOCHS_OVERRIDE, "ATTACK_EPOCHS_OVERRIDE"
        elif attack.history.get("epoch"):
            epochs, source = len(attack.history["epoch"]), "checkpoint history (epochs up to best-ASR checkpoint)"
        else:
            epochs, source = None, "unknown (set ATTACK_EPOCHS_OVERRIDE)"

        cda0, asr0 = attack.evaluate(self.test_loader, defense=DummyNoDefense())
        runtime = ProtoGuardRuntime(num_classes=Config.NUM_CLASSES)
        print("\n  Fitting ProtoGuard-SL on labeled training data (class prototypes)...")
        runtime.fit(attack.client_model, self.calib_loader, self.device)
        cda, asr = attack.evaluate(self.test_loader, defense=runtime)
        print(f"  No Defense      -- CDA: {cda0:.2f}%  ASR: {asr0:.2f}%")
        print(f"  ProtoGuard-SL   -- CDA: {cda:.2f}%  ASR: {asr:.2f}%  (ASR drop: {asr0 - asr:.2f} pts)")
        shared = {"attack_epochs": epochs, "attack_epochs_detail": source,
                  "attack_iterations": epochs * len(self.train_loader) if epochs else float('nan'),
                  "iteration_unit": "training steps (epochs x train batches)",
                  "eval_iterations": len(self.test_loader)}
        rows = self.pair(attack_key, shared, {"acc": cda0, "metric": asr0}, {"acc": cda, "metric": asr},
                         runtime, "calibration + inference-time prototype replacement", runtime.fit_passes)
        rows[1].update({"mode": mode, "no_defense_cda": cda0, "no_defense_asr": asr0,
                        "protoguard_cda": cda, "protoguard_asr": asr})
        return rows

    def clean_model(self):
        if self._clean is not None:
            return self._clean
        client, server = build_split_models(self.model_name, self.in_channels, self.cut, Config.NUM_CLASSES)
        client, server = client.to(self.device), server.to(self.device)
        size = 28 if Config.DATASET == "MNIST" else 32
        with torch.no_grad():
            server(client(torch.zeros(2, self.in_channels, size, size, device=self.device)))

        own = f"{Config.SAVE_DIR}/protoguard_clean_{self.model_name.lower()}_cut{self.cut}_{Config.DATASET}.pth"
        candidates = [f"{Config.SAVE_DIR}/best_{CLEAN_CHECKPOINT_TAGS[self.model_name]}_{Config.DATASET}.pth",
                      f"{Config.SAVE_DIR}/best_{Config.MODEL_NAME.lower()}_sl_{Config.DATASET}.pth", own]
        for path in candidates:
            if not os.path.exists(path):
                continue
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            if self.model_name != "Vanilla" and ckpt.get("cut_layer", self.cut) != self.cut:
                print(f"  [skip] {path} was trained at cut {ckpt.get('cut_layer')}, current cut is {self.cut}")
                continue
            try:
                client.load_state_dict(ckpt["client_state"])
                server.load_state_dict(ckpt["server_state"])
            except RuntimeError:
                print(f"  [skip] {path} does not match this model / cut layer")
                continue
            epochs = ckpt.get("trained_epochs", ckpt["epoch"] + 1 if "epoch" in ckpt else None)
            print(f"  [✓] clean victim model loaded: {path}")
            self._clean = (client, server, epochs, path)
            return self._clean

        print(f"  [!] no clean checkpoint for {self.model_name} cut {self.cut}: training {CLEAN_EPOCHS} epoch(s)")
        set_seed()
        opt_c = optim.Adam(client.parameters(), lr=Config.LEARNING_RATE)
        opt_s = optim.Adam(server.parameters(), lr=Config.LEARNING_RATE)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(CLEAN_EPOCHS):
            client.train(); server.train()
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                opt_c.zero_grad(); opt_s.zero_grad()
                loss_fn(server(client(inputs)), labels).backward()
                opt_c.step(); opt_s.step()
            print(f"    clean epoch {epoch + 1}/{CLEAN_EPOCHS} | test acc "
                  f"{test_accuracy(client, server, self.test_loader, self.device):.2f}%")
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
        torch.save({"client_state": client.state_dict(), "server_state": server.state_dict(),
                    "cut_layer": self.cut, "trained_epochs": CLEAN_EPOCHS, "dataset": Config.DATASET}, own)
        self._clean = (client, server, CLEAN_EPOCHS, own)
        return self._clean

    def run_reconstruction(self, attack_key, attack_fn, attack_epochs, iterations, unit):
        client, server, victim_epochs, source = self.clean_model()
        print(f"\n  >> {attack_key}: baseline (no defense)")
        set_seed()
        m0 = attack_fn(client, LabelHolder(), None)
        acc0 = test_accuracy(client, server, self.test_loader, self.device)

        print(f"\n  >> {attack_key}: ProtoGuard-SL on the observed smashed data")
        runtime = ProtoGuardRuntime(num_classes=Config.NUM_CLASSES)
        runtime.fit(client, self.calib_loader, self.device)
        set_seed()
        m1 = attack_fn(client, LabelHolder(), runtime)
        calls, replaced = runtime.calls, runtime.replaced_pct
        acc1 = test_accuracy(client, server, self.test_loader, self.device, runtime)
        shared = {"attack_epochs": attack_epochs, "attack_iterations": iterations, "iteration_unit": unit,
                  "victim_model_epochs": victim_epochs, "victim_checkpoint": source}
        return self.pair(attack_key, shared,
                         {"acc": acc0, "metric": m0["ssim"], "psnr": m0["psnr"], "mse": m0["mse"]},
                         {"acc": acc1, "metric": m1["ssim"], "psnr": m1["psnr"], "mse": m1["mse"]},
                         runtime, "inference: filters the smashed data the attacker observes", runtime.fit_passes,
                         calls=calls, replaced_pct=replaced)

    def run_whitebox(self):
        from all_attacks.attacks_whitebox import WhiteBoxInversionAttack, AttackMetricsTracker

        def attack_fn(client, holder, runtime):
            attacker = WhiteBoxInversionAttack(client_model=client, dataset=Config.DATASET,
                                               iterations=WHITEBOX_ITERS, lr=1e-2)
            tracker, done, mses = AttackMetricsTracker(), 0, []
            for inputs, labels in self.test_loader:
                if done >= RECON_IMAGES:
                    break
                inputs = inputs[:RECON_IMAGES - done].to(self.device)
                labels = labels[:RECON_IMAGES - done].to(self.device)
                with torch.no_grad():
                    smashed = client(inputs)
                    if runtime is not None:
                        smashed = runtime.protect(smashed, labels)
                recon = attacker.reconstruct(smashed, inputs.shape)
                orig = denormalize(inputs)
                tracker.log_batch(orig, recon)
                mses.extend(((orig - recon) ** 2).flatten(1).mean(1).tolist())
                done += inputs.shape[0]
            s = tracker.get_summary()
            return {"ssim": float(s["mean_ssim"]), "psnr": float(s["mean_psnr"]), "mse": float(np.mean(mses))}

        return self.run_reconstruction("whitebox", attack_fn, 0, WHITEBOX_ITERS * RECON_IMAGES,
                                       "inversion steps (steps per image x images)")

    def run_unsplit(self):
        from all_attacks.attack_unsplit import UnSplitAttack
        n_batches = max(1, RECON_IMAGES // Config.BATCH_SIZE)
        main_iters = getattr(Config, "unsplit_main_iters", None)
        if main_iters is not None:
            inner = getattr(Config, "unsplit_input_iters", 0) + getattr(Config, "unsplit_model_iters", 0)
            iterations, unit = main_iters * inner * n_batches, \
                "optimisation steps (main iters x (input + model iters) x batches, from config.py)"
        else:
            iterations, unit = UNSPLIT_STEPS * n_batches, "inversion steps x batches"
        clone = lambda: build_split_models(self.model_name, self.in_channels, self.cut, Config.NUM_CLASSES)[0]

        def attack_fn(client, holder, runtime):
            target = client if runtime is None else ProtectedClient(client, runtime, holder)
            attacker = call_with_supported_args(UnSplitAttack, client_model=target, in_channels=self.in_channels,
                                                clone_builder=clone)
            loader = TappedLoader(self.test_loader, holder)
            m = call_with_supported_args(attacker.run_attack, loader, num_batches=n_batches,
                                         inversion_steps=UNSPLIT_STEPS)
            return {"ssim": metric(m, "ssim", "mean_ssim"), "psnr": metric(m, "psnr", "mean_psnr"),
                    "mse": metric(m, "mse", "mean_mse")}

        return self.run_reconstruction("unsplit", attack_fn, 0, iterations, unit)

    def run_ae_decoder(self):
        from all_attacks.ae_decoder_attack import run_ae_decoder_attack
        train_batches, test_batches = 80, max(1, RECON_IMAGES // 32 + 1)

        def attack_fn(client, holder, runtime):
            target = client if runtime is None else ProtectedClient(client, runtime, holder)
            summary, _, _ = run_ae_decoder_attack(target, TappedLoader(self.train_loader, holder),
                                                  TappedLoader(self.test_loader, holder), self.device,
                                                  Config.DATASET, ae_epochs=AE_EPOCHS,
                                                  collect_train_batches=train_batches,
                                                  collect_test_batches=test_batches,
                                                  label="AE Decoder [" + ("ProtoGuard" if runtime else "no defense")
                                                        + "]")
            return {"ssim": summary["mean_ssim"], "psnr": summary["mean_psnr"], "mse": summary["mean_mse"]}

        pairs = min(train_batches, len(self.train_loader)) * Config.BATCH_SIZE
        return self.run_reconstruction("ae_decoder", attack_fn, AE_EPOCHS,
                                       AE_EPOCHS * int(np.ceil(0.9 * pairs / 32)),
                                       "decoder training steps (AE epochs x AE batches)")

    def run_fsha(self):
        from all_attacks.fsha_attack import FSHAAttack, compute_mse, compute_psnr, compute_ssim
        set_seed()
        client = build_split_models(self.model_name, self.in_channels, self.cut, Config.NUM_CLASSES)[0]
        fsha = FSHAAttack(client_model=client.to(self.device), in_channels=self.in_channels, dataset=Config.DATASET,
                          pilot_builder=lambda: build_split_models(self.model_name, self.in_channels, self.cut,
                                                                   Config.NUM_CLASSES)[0],
                          critic_iters=FSHA_CRITIC_ITERS)
        private = DataLoader(self.train_loader.dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
        public = DataLoader(self.test_loader.dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
        fsha.hijack(private, public, epochs=FSHA_EPOCHS)

        @torch.no_grad()
        def reconstruct(runtime):
            fsha.client_model.eval(); fsha.decoder.eval()
            psnr, ssim, mse, seen = [], [], [], 0
            for images, labels in self.train_loader:
                if seen >= RECON_IMAGES:
                    break
                images = images[:RECON_IMAGES - seen].to(self.device)
                labels = labels[:RECON_IMAGES - seen].to(self.device)
                smashed = fsha.client_model(images)
                if runtime is not None:
                    smashed = runtime.protect(smashed, labels)
                rec = fsha.decoder(smashed).clamp(0, 1)
                orig = denormalize(images)
                for i in range(images.shape[0]):
                    psnr.append(compute_psnr(orig[i], rec[i]))
                    ssim.append(compute_ssim(orig[i].unsqueeze(0), rec[i].unsqueeze(0)))
                    mse.append(compute_mse(orig[i], rec[i]))
                seen += images.shape[0]
            return {"ssim": float(np.mean(ssim)), "psnr": float(np.mean(psnr)), "mse": float(np.mean(mse))}

        m0 = reconstruct(None)
        runtime = ProtoGuardRuntime(num_classes=Config.NUM_CLASSES)
        runtime.fit(fsha.client_model, self.calib_loader, self.device)
        m1 = reconstruct(runtime)
        shared = {"attack_epochs": FSHA_EPOCHS,
                  "attack_iterations": FSHA_EPOCHS * len(private) * (2 + FSHA_CRITIC_ITERS),
                  "iteration_unit": "optimisation steps (epochs x batches x (pilot + critic iters + hijack))"}
        return self.pair("fsha", shared,
                         {"acc": float('nan'), "metric": m0["ssim"], "psnr": m0["psnr"], "mse": m0["mse"]},
                         {"acc": float('nan'), "metric": m1["ssim"], "psnr": m1["psnr"], "mse": m1["mse"]},
                         runtime, "inference: filters the hijacked client's smashed data", runtime.fit_passes)

    def run(self, attack_key):
        if attack_key == "poison_client":
            return self.run_one("client")
        if attack_key == "poison_server":
            return self.run_one("server")
        return getattr(self, f"run_{attack_key}")()


if __name__ == "__main__":
    if QUICK_TEST:
        LEAK_EPOCHS = VILLAIN_WARMUP = VILLAIN_INFERENCE = VILLAIN_INJECTION = CLEAN_EPOCHS = FSHA_EPOCHS = 1
        RECON_IMAGES, WHITEBOX_ITERS, UNSPLIT_STEPS, AE_EPOCHS, FSHA_CRITIC_ITERS, CALIB_BATCHES = 4, 20, 20, 2, 1, 10

    print("=" * 60)
    print("  PROTOGUARD-SL DEFENSE EXPERIMENT")
    print(f"  Dataset : {Config.DATASET}")
    print(f"  Models  : {model_from_config()}")
    print(f"  Cuts    : {Config.CUT_LAYER}")
    print(f"  Modes   : {MODES_TO_RUN}")
    print(f"  alpha   : {ALPHA} (paper default)")
    print("=" * 60)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"  Device  : {device}\n")
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    exp = Experiment(device)

    print("=" * 78)
    print("  PROTOGUARD-SL vs ALL ATTACKS")
    print(f"  Dataset  : {Config.DATASET}")
    print(f"  Model    : {exp.model_name}  (Config.MODEL_NAME = {Config.MODEL_NAME})")
    print(f"  Cut layer: {exp.cut_label}")
    print(f"  Device   : {device}")
    print(f"  alpha    : {ALPHA} (paper default) | training-attack stage: {TRAINING_STAGE}")
    print(f"  Attacks  : {ATTACKS_TO_RUN}")
    print("=" * 78)

    all_rows = []
    output_path = f"{Config.RESULTS_DIR}/protoguard_sl_all_attacks_{Config.DATASET}.csv"

    for attack_key in ATTACKS_TO_RUN:
        if attack_key.startswith("poison_") and attack_key.split("_")[1] not in MODES_TO_RUN:
            continue
        print("\n" + "#" * 78)
        print(f"#   PROTOGUARD-SL vs {attack_key} -- {ATTACKS[attack_key][0]}")
        print(f"#   MODEL: {exp.model_name} | CUT LAYER: {exp.cut_label} | {Config.DATASET}")
        print("#" * 78)
        try:
            rows = exp.run(attack_key)
        except Exception as exc:
            traceback.print_exc()
            r = exp.row(attack_key, f"protoguard_a{ALPHA:g}")
            r.update({"verdict": NOT_APPLICABLE, "reason": f"run failed: {type(exc).__name__}: {exc}"})
            rows = [finalize_row(r)]
        for r in rows:
            if r['verdict'] != BASELINE:
                print(f"\n  [run info] attack={attack_key} | cut={r['cut_layer']} | attack epochs="
                      f"{r.get('attack_epochs')} | defense epochs={r.get('defense_epochs')} | iterations="
                      f"{r.get('attack_iterations')} | defense calls={r.get('defense_calls')}")
                print(f"  Result: {r['defense_working']} -- {r['reason']}")
        all_rows.extend(rows)
        save_results(all_rows, output_path)

    print_summary_table(all_rows, f"PROTOGUARD-SL vs ALL ATTACKS -- {exp.model_name} | cut {exp.cut_label} | "
                                  f"{Config.DATASET}")
    save_results(all_rows, output_path)
    print(f"\n  Saved -> {output_path}")

    results = [{"model": r["model"], "cut_layer": Config.CUT_LAYER, "mode": r["mode"],
                "no_defense_cda": r["no_defense_cda"], "no_defense_asr": r["no_defense_asr"],
                "protoguard_cda": r["protoguard_cda"], "protoguard_asr": r["protoguard_asr"]}
               for r in all_rows if "protoguard_asr" in r]
    if not results:
        print("\n  No poisoned checkpoints were found -- nothing to evaluate.")
        print("  Run defence_runner/run_backdoor_poison.py first (same cut layer).")
    else:
        df = pd.DataFrame(results)
        output_path = f"{Config.RESULTS_DIR}/protoguard_sl_defense_evaluation_{Config.DATASET}.csv"
        if os.path.exists(output_path):
            old = pd.read_csv(output_path)
            if "cut_layer" in old.columns:
                df = pd.concat([old, df]).drop_duplicates(
                    subset=["model", "cut_layer", "mode"], keep="last")
        df = df.sort_values(["cut_layer", "model", "mode"])
        df.to_csv(output_path, index=False)

        print("\n" + "=" * 100)
        print(f"   PROTOGUARD-SL DEFENSE EVALUATION -- {Config.DATASET}")
        print("=" * 100)
        print(f"{'Model':<12} {'Cut':>4} {'Mode':<8} {'No-Def CDA':>11} {'No-Def ASR':>11} "
              f"{'ProtoGuard CDA':>15} {'ProtoGuard ASR':>15}")
        print("-" * 100)
        for r in df.to_dict("records"):
            print(f"  {r['model']:<10} {int(r['cut_layer']):>4} {r['mode']:<8} "
                  f"{r['no_defense_cda']:>11.2f} {r['no_defense_asr']:>11.2f} "
                  f"{r['protoguard_cda']:>15.2f} {r['protoguard_asr']:>15.2f}")
        print("=" * 100)
        print(f"\nSaved raw data -> {output_path}")