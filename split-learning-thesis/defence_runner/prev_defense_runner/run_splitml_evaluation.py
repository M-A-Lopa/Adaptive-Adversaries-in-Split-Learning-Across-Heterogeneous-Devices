import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from config import Config
from dataset import DatasetLoader
from all_defences.splitml_defense import SplitMLDefense, OPENFHE_AVAILABLE
from all_split_learning.splitml_federated_learning import SplitMLTrainer, build_splitml_clients
from all_attacks.ae_decoder_attack import run_ae_decoder_attack
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel
from all_attacks.attacks_whitebox import WhiteBoxInversionAttack, AttackMetricsTracker


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

NUM_CLIENTS        = 3
ROUNDS             = 10
RUN_STANDALONE     = True
CONSENSUS_LAMBDA   = 0.10
CONSENSUS_BATCHES  = None
EXTRA_DP_SIGMA     = 0.0

RUN_WHITEBOX, RUN_AE = True, True
MAX_IMAGES, ITERATIONS, AE_EPOCHS = 32, 1000, 50
RUN_UNSPLIT   = True
UNSPLIT_STEPS = 1000

POISON_TARGET  = 0
POISON_RATE    = 0.05
POISON_PATCH   = 4
POISON_TRIGGER = 1.0

QUICK_TEST = False


def attack_suite(client_model, view_fn, train_loader, test_loader, device, tag):
    row = {}
    if RUN_WHITEBOX:
        s = run_whitebox_vs_view(client_model, view_fn, test_loader, device, Config.DATASET, MAX_IMAGES, ITERATIONS)
        row.update(wb_psnr=s['mean_psnr'], wb_ssim=s['mean_ssim'])
    if RUN_AE:
        s, _, _ = run_ae_decoder_attack(DefendedClient(client_model, view_fn), train_loader, test_loader, device,
                                        Config.DATASET, ae_epochs=AE_EPOCHS, label=f'AE Decoder vs {tag}')
        row.update(ae_psnr=s['mean_psnr'], ae_ssim=s['mean_ssim'])
    if RUN_UNSPLIT:
        ic = 1 if Config.DATASET == 'MNIST' else 3
        s = run_unsplit_vs_view(client_model, view_fn, test_loader, ic, lambda: build_client(ic))
        row.update(us_psnr=s['psnr'], us_ssim=s['ssim'])
    return row


import math
import inspect
import random
import traceback

ATTACKS_TO_RUN = ["label_leakage", "villain", "poison_client", "poison_server",
                  "whitebox", "unsplit", "ae_decoder", "fsha"]

ATTACK_INFO = {
    "label_leakage": ("Label Leakage (norm + cosine gradient scoring)", "label_inference"),
    "villain":       ("VILLAIN backdoor (label inference + embedding trigger)", "backdoor"),
    "poison_client": ("Backdoor poisoning, client attacker (BadNets patch)", "backdoor"),
    "poison_server": ("Backdoor poisoning, server attacker (surrogate client)", "backdoor"),
    "whitebox":      ("White-box model inversion", "reconstruction"),
    "unsplit":       ("UnSplit model inversion", "reconstruction"),
    "ae_decoder":    ("AE decoder inversion", "reconstruction"),
    "fsha":          ("FSHA feature-space hijacking", "reconstruction"),
}

RECON_KEYS = {"whitebox": "wb", "unsplit": "us", "ae_decoder": "ae", "fsha": "fsha"}

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


def judge(family, base, dfd, psnr_base=None, psnr_dfd=None):
    if _missing(base) or _missing(dfd):
        return NOT_APPLICABLE, "metric unavailable (no undefended baseline to compare with)"
    t = THRESHOLDS
    if family == "label_inference":
        thr, drop, name, fmt = t['leak_auc_success'], t['leak_auc_partial_drop'], "leak AUC", ".3f"
    elif family == "backdoor":
        thr, drop, name, fmt = t['asr_success'], t['asr_partial_drop'], "ASR %", ".1f"
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


def run_unsplit_vs_view(client_model, view_fn, test_loader, in_channels, clone_builder):
    from all_attacks.attack_unsplit import UnSplitAttack
    n_batches = max(1, MAX_IMAGES // Config.BATCH_SIZE)
    attacker = call_with_supported_args(UnSplitAttack, client_model=DefendedClient(client_model, view_fn),
                                        in_channels=in_channels, clone_builder=clone_builder)
    m = call_with_supported_args(attacker.run_attack, test_loader, num_batches=n_batches,
                                 inversion_steps=UNSPLIT_STEPS)
    return {'psnr': _metric(m, 'psnr', 'mean_psnr'), 'ssim': _metric(m, 'ssim', 'mean_ssim')}


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


SPLITML_NOT_APPLICABLE = {
    "label_leakage": "no cut-layer gradients are exchanged and labels never leave a client (each client trains "
                     "its whole model locally)",
    "villain":       "no cut-layer gradients or smashed data are exchanged during training, so there is no signal "
                     "to infer labels from or embedding channel to inject a trigger into",
    "poison_server": "the server only averages CKKS ciphertexts of the shared top layers and never runs a model "
                     "on client data",
    "fsha":          "no server->client gradient channel exists, so the server cannot hijack client training",
}


class PoisonedLoader:
    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        from all_attacks.backdoor_poison_attack import denormalize as p_den, normalize as p_norm, apply_trigger_patch
        for x, y in self.loader:
            y = y.clone()
            eligible = (y != POISON_TARGET).nonzero(as_tuple=True)[0]
            if len(eligible):
                n = min(max(1, int(POISON_RATE * len(y))), len(eligible))
                idx = eligible[torch.randperm(len(eligible))[:n]]
                raw = p_den(x, Config.DATASET)
                raw[idx] = apply_trigger_patch(raw[idx], POISON_PATCH, POISON_TRIGGER)
                x = p_norm(raw, Config.DATASET)
                y[idx] = POISON_TARGET
            yield x, y


@torch.no_grad()
def backdoor_asr(model, test_loader, device):
    from all_attacks.backdoor_poison_attack import denormalize as p_den, normalize as p_norm, apply_trigger_patch
    model.eval()
    hit = total = 0
    for x, y in test_loader:
        rows = y != POISON_TARGET
        if rows.sum() == 0:
            continue
        trig = p_norm(apply_trigger_patch(p_den(x[rows], Config.DATASET), POISON_PATCH, POISON_TRIGGER), Config.DATASET)
        hit += model(trig.to(device)).argmax(1).eq(POISON_TARGET).sum().item()
        total += int(rows.sum())
    return 100.0 * hit / max(total, 1)


def run_splitml_poison_client(in_channels, device, train_dataset, test_loader):
    torch.manual_seed(0)
    tr = SplitMLTrainer(build_splitml_clients(NUM_CLIENTS, in_channels, Config.NUM_CLASSES, device),
                        train_dataset, test_loader, ROUNDS, collaborative=True, extra_dp_sigma=EXTRA_DP_SIGMA,
                        tag="splitml_poisoned_client1")
    tr.loaders[0] = PoisonedLoader(tr.loaders[0])
    tr.train()
    asr = [backdoor_asr(c, test_loader, device) for c in tr.clients]
    acc = [tr.evaluate_client(k) for k in range(tr.K)]
    return asr, acc, len(tr.loaders[0])


if __name__ == "__main__":
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    if QUICK_TEST:
        ROUNDS, MAX_IMAGES, ITERATIONS, AE_EPOCHS, UNSPLIT_STEPS = 1, 4, 20, 2, 20
    print(f"Using execution device: {device}")
    print(f"CKKS backend: {'OpenFHE (real multi-key CKKS)' if OPENFHE_AVAILABLE else 'SimulatedCKKS — pip install openfhe for real FHE'}")
    train_loader, test_loader = DatasetLoader(dataset_name=Config.DATASET).get_loaders()
    in_channels = 1 if Config.DATASET == 'MNIST' else 3
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    torch.manual_seed(0)
    clients = build_splitml_clients(NUM_CLIENTS, in_channels, Config.NUM_CLASSES, device)
    trainer = SplitMLTrainer(clients, train_loader.dataset, test_loader, ROUNDS, collaborative=True,
                             extra_dp_sigma=EXTRA_DP_SIGMA)
    fed_row = trainer.train()

    standalone_row = None
    if RUN_STANDALONE:
        torch.manual_seed(0)
        solo = SplitMLTrainer(build_splitml_clients(NUM_CLIENTS, in_channels, Config.NUM_CLASSES, device),
                              train_loader.dataset, test_loader, ROUNDS, collaborative=False)
        standalone_row = solo.train()

    print("\n" + "=" * 60 + f"\n  CONSENSUS INFERENCE (Alg. 5), lambda = {CONSENSUS_LAMBDA}\n" + "=" * 60)
    consensus = trainer.evaluate_consensus(lam=CONSENSUS_LAMBDA, max_batches=CONSENSUS_BATCHES)
    pd.DataFrame(consensus).to_csv(
        f"{Config.RESULTS_DIR}/splitml_consensus_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv", index=False)

    results = []
    ckpt = vanilla_checkpoint_path()
    if os.path.exists(ckpt):
        base_client = build_client(in_channels).to(device)
        base_client.load_state_dict(torch.load(ckpt, map_location=device)['client_state'])
        print("\n" + "=" * 60 + "\n  BASELINE (NO DEFENSE) — vanilla SL server sees smashed data\n" + "=" * 60)
        results.append({'setting': 'Vanilla SL (No Defense)',
                        **attack_suite(base_client, lambda z: z, train_loader, test_loader, device, 'no_defense_splitml')})

    top = trainer.clients[0].top
    for collusion in (False, True):
        d = SplitMLDefense(collusion=collusion)
        print("\n" + "=" * 60 + f"\n  {d}\n" + "=" * 60)
        label = "SplitML peer holds SK' (collusion bound)" if collusion else "SplitML curious peer (no SK')"
        results.append({'setting': label, **attack_suite(top, d.protect, train_loader, test_loader, device,
                                                           'splitml_collusion' if collusion else 'splitml_peer')})

    if not any(r['setting'].startswith('Vanilla SL') for r in results):
        print("\n" + "=" * 60 + "\n  BASELINE (NO DEFENSE) -- curious peer sees the raw activations of the top\n"
              "  (no vanilla checkpoint for this model)\n" + "=" * 60)
        results.insert(0, {'setting': 'No Defense (peer sees raw activations)',
                           **attack_suite(top, lambda z: z, train_loader, test_loader, device, 'no_defense_splitml_raw')})

    for r in results:
        r.update(server_inversion='n/a (server sees only ciphertexts)', fsha='n/a (no server->client gradient channel)')
    df = pd.DataFrame(results)
    out = f"{Config.RESULTS_DIR}/splitml_defense_evaluation_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 100)
    print(f" THESIS TABLE — SplitML (Trivedi et al.) — {Config.MODEL_NAME} / {Config.DATASET}")
    print("=" * 100)
    for k in range(NUM_CLIENTS):
        s = f"  Client {k+1} (head variant {k}): federated {fed_row[f'client{k+1}_test_acc']:.2f}%"
        if standalone_row:
            s += f" | standalone {standalone_row[f'client{k+1}_test_acc']:.2f}%"
        print(s)
    for c in consensus:
        print(f"  Requester C{c['requester']}: own {c['own_acc']:.2f}% | TL {c['TL_acc']:.2f}% | TP {c['TP_acc']:.2f}%")
    print(f"  Federation overhead: {trainer.fed_time:.1f}s over {ROUNDS} rounds")
    print("-" * 100)
    cols = [c for c in ['wb_psnr', 'wb_ssim', 'us_psnr', 'us_ssim', 'ae_psnr', 'ae_ssim'] if c in df]
    print(f"{'Setting':<44}" + "".join(f"{c:<12}" for c in cols))
    for r in results:
        print(f"{r['setting']:<44}" + "".join(f"{r.get(c, float('nan')):<12.4f}" for c in cols))
    print("  FSHA / server-side inversion: n/a — the SplitML server never receives smashed data")
    print("=" * 100)
    print(f"\nSaved raw data → {out}")

    model_label = Config.MODEL_NAME
    cut_label = (Config.CUT_LAYER if Config.MODEL_NAME in ("KAGN", "PyramidCNN") else "fixed")
    acc0 = float('nan')
    if os.path.exists(ckpt):
        acc0 = torch.load(ckpt, map_location='cpu', weights_only=False).get('best_acc', float('nan'))
    fed_acc = float(np.mean([fed_row[f'client{k+1}_test_acc'] for k in range(NUM_CLIENTS)]))
    all_out = f"{Config.RESULTS_DIR}/splitml_all_attacks_{Config.DATASET}.csv"
    stage = "training: whole model trained locally, shared top averaged under multi-key CKKS"

    print("\n" + "#" * 78)
    print("#   SplitML vs ALL ATTACKS")
    print(f"#   MODEL: {model_label} | CUT LAYER: {cut_label} | {Config.DATASET}")
    print("#" * 78)
    all_rows = []
    for r in results[1:]:
        defense_tag = "splitml_collusion" if "collusion" in r['setting'] else "splitml_peer"
        for attack_key in ATTACKS_TO_RUN:
            if attack_key in SPLITML_NOT_APPLICABLE and ATTACK_INFO[attack_key][1] != "reconstruction":
                all_rows.append(na_row(attack_key, defense_tag, model_label, cut_label,
                                       SPLITML_NOT_APPLICABLE[attack_key]))
        all_rows += recon_rows(results[0], r, defense_tag, model_label, cut_label, len(train_loader), ROUNDS,
                               stage + (" (attacker peer holds SK')" if "collusion" in r['setting'] else ""),
                               acc0, fed_acc, {k: v for k, v in SPLITML_NOT_APPLICABLE.items() if k == "fsha"})

    if "poison_client" in ATTACKS_TO_RUN:
        print("\n" + "#" * 78)
        print("#   SplitML vs poison_client -- client 1 is a malicious federation member (poisoned shard)")
        print("#" * 78)
        try:
            asr, acc, n_batches = run_splitml_poison_client(in_channels, device, train_loader.dataset, test_loader)
            honest_clean = float(np.mean([fed_row[f'client{k+1}_test_acc'] for k in range(1, NUM_CLIENTS)]))
            row = finalize_row(summary_row(
                "poison_client", "splitml_federation", model_label, cut_label,
                attack_epochs=ROUNDS, defense_epochs=ROUNDS, defense_stage=stage,
                attack_iterations=ROUNDS * n_batches, iteration_unit="training steps of the malicious client",
                defense_calls=ROUNDS, accuracy_no_defense=honest_clean,
                accuracy_with_defense=float(np.mean(acc[1:])),
                metric_no_defense=asr[0], metric_with_defense=float(np.mean(asr[1:])),
                note=(f"Before = backdoor on the malicious client's own model, After = mean backdoor on the "
                      f"{NUM_CLIENTS - 1} honest clients' models (per client: "
                      + ", ".join(f"C{k+1} {a:.1f}%" for k, a in enumerate(asr)) + ")")))
        except Exception as exc:
            traceback.print_exc()
            row = na_row("poison_client", "splitml_federation", model_label, cut_label,
                         f"run failed: {type(exc).__name__}: {exc}")
        all_rows.append(row)

    order = {a: i for i, a in enumerate(ATTACKS_TO_RUN)}
    all_rows.sort(key=lambda r: order.get(r["attack"], 99))
    for r in all_rows:
        print_run_info(r)
    print_summary_table(all_rows, f"SplitML vs ALL ATTACKS -- {model_label} | cut {cut_label} | {Config.DATASET}")
    print_defense_attack_matrix(all_rows)
    save_results(all_rows, all_out)
    print(f"\n  Saved -> {all_out}")
