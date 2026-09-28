import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from config import Config
from dataset import DatasetLoader
from all_defences.r3elu_defense import R3eLUMechanism
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
        super().__init__(*args, **kw)
        self.view_fn = view_fn if view_fn is not None else (lambda z: z)
        self.tag = tag

    def _observed(self, smashed):
        return smashed + (self.view_fn(smashed) - smashed).detach()

    def hijack(self, private_loader, public_loader, epochs=5):
        print(f"\n  Running FSHA against defense view '{self.tag}' for {epochs} epoch(s)...")
        self.client_model.train(); self.pilot.train(); self.decoder.train(); self.critic.train()
        for epoch in range(epochs):
            pub_iter = iter(public_loader)

            def next_pub():
                nonlocal pub_iter
                try:
                    x, _ = next(pub_iter)
                except StopIteration:
                    pub_iter = iter(public_loader)
                    x, _ = next(pub_iter)
                return x.to(self.device)

            rc = rh = rr = 0.0
            n = 0
            for priv, _ in tqdm(private_loader, desc=f"  Hijack [{epoch+1}/{epochs}]", leave=False):
                priv = priv.to(self.device)
                pub = next_pub()
                self.pilot_optimizer.zero_grad()
                recon_loss = self.mse(self.decoder(self.pilot(pub)), pub)
                recon_loss.backward()
                self.pilot_optimizer.step()

                with torch.no_grad():
                    fake = self.view_fn(self.client_model(priv))
                for _ in range(self.critic_iters):
                    with torch.no_grad():
                        real = self.pilot(next_pub())
                    self.critic_optimizer.zero_grad()
                    n = min(real.shape[0], fake.shape[0])
                    gp = gradient_penalty(self.critic, real[:n], fake[:n], self.device)
                    critic_loss = self.critic(fake[:n]).mean() - self.critic(real[:n]).mean() + self.gp_lambda * gp
                    critic_loss.backward()
                    self.critic_optimizer.step()

                self.client_optimizer.zero_grad()
                hijack_loss = -self.critic(self._observed(self.client_model(priv))).mean()
                hijack_loss.backward()
                self.client_optimizer.step()
                rc += critic_loss.item(); rh += hijack_loss.item(); rr += recon_loss.item(); n += 1
            print(f"  Epoch {epoch+1:2d}/{epochs} | Critic: {rc/n:.4f} | Hijack: {rh/n:.4f} | Pilot-Recon: {rr/n:.4f}")

    def reconstruct(self, test_loader, num_images=32):
        self.client_model.eval(); self.decoder.eval()
        psnr, ssim, mse = [], [], []
        vis_o = vis_r = None
        seen = 0
        with torch.no_grad():
            for images, _ in test_loader:
                if seen >= num_images:
                    break
                images = images[:num_images - seen].to(self.device)
                rec = self.decoder(self.view_fn(self.client_model(images))).clamp(0, 1)
                orig = fsha_denormalize(images, self.dataset)
                for i in range(images.shape[0]):
                    psnr.append(compute_psnr(orig[i], rec[i]))
                    ssim.append(compute_ssim(orig[i].unsqueeze(0), rec[i].unsqueeze(0)))
                    mse.append(compute_mse(orig[i], rec[i]))
                if vis_o is None:
                    vis_o, vis_r = orig[:8].cpu(), rec[:8].cpu()
                seen += images.shape[0]
        summary = {'mse': float(np.mean(mse)), 'psnr': float(np.mean(psnr)), 'ssim': float(np.mean(ssim))}
        print(f"  FSHA vs {self.tag}: PSNR {summary['psnr']:.2f} dB | SSIM {summary['ssim']:.4f} | MSE {summary['mse']:.5f}")
        self._save_visualization(vis_o, vis_r, tag=self.tag)
        return summary


def run_fsha_vs_view(view_fn, in_channels, private_loader, public_loader, tag, epochs=5, num_images=32,
                     client_builder=None):
    builder = client_builder or (lambda: build_client(in_channels))
    attack = DefendedFSHAAttack(client_model=builder(), in_channels=in_channels, dataset=Config.DATASET,
                                pilot_builder=builder, critic_iters=5, view_fn=view_fn, tag=tag)
    attack.hijack(private_loader, public_loader, epochs=epochs)
    return attack.reconstruct(private_loader, num_images=num_images)

EPSILON_VALUES   = [0.1, 0.5, 1.0, 2.0, 4.0]
C_CLIP           = 10.0
K_FRACTION       = 0.5
PROTECT_BACKWARD = True
DYNAMIC_BUDGET   = True
EPOCHS_PER_EPS   = 10

RUN_WHITEBOX, RUN_AE, RUN_FSHA = True, True, True
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


def attack_suite(client_model, view_fn, train_loader, test_loader, device, in_channels, tag):
    row = {}
    if RUN_WHITEBOX:
        s = run_whitebox_vs_view(client_model, view_fn, test_loader, device, Config.DATASET, MAX_IMAGES, ITERATIONS)
        row.update(wb_psnr=s['mean_psnr'], wb_ssim=s['mean_ssim'])
    if RUN_AE:
        s, _, _ = run_ae_decoder_attack(DefendedClient(client_model, view_fn), train_loader, test_loader, device,
                                        Config.DATASET, ae_epochs=AE_EPOCHS, label=f'AE Decoder vs {tag}')
        row.update(ae_psnr=s['mean_psnr'], ae_ssim=s['mean_ssim'])
    if RUN_FSHA:
        s = run_fsha_vs_view(view_fn, in_channels, train_loader, test_loader, tag=tag, epochs=FSHA_EPOCHS)
        row.update(fsha_psnr=s['psnr'], fsha_ssim=s['ssim'])
    if RUN_UNSPLIT:
        s = run_unsplit_vs_view(client_model, view_fn, test_loader, in_channels, lambda: build_client(in_channels))
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


from all_defences.r3elu_defense import R3eLU


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


def r3elu_villain(eps, n_features, base_dataset, test_loader, in_channels):
    from all_attacks.villain_backdoor_attack import VILLAINBackdoorAttack, build_indexed_loader
    _seed()
    attack = VILLAINBackdoorAttack(client_model=build_client(in_channels),
                                   server_model=build_server_n(in_channels, Config.NUM_CLASSES),
                                   base_dataset=base_dataset, dataset=Config.DATASET, num_classes=Config.NUM_CLASSES,
                                   target_label=VILLAIN_TARGET, poison_rate=VILLAIN_POISON,
                                   candidates_per_batch=VILLAIN_CANDIDATES)
    wrapper = None
    if eps is not None:
        wrapper = R3eLUProtectedServer(attack.server_model, make_mech(eps, n_features, attack.device))
        attack.server_model = wrapper
    loader = build_indexed_loader(base_dataset, batch_size=VILLAIN_BATCH, shuffle=True)
    attack.warmup(loader, epochs=VILLAIN_WARMUP)
    attack.infer_labels(loader, epochs=VILLAIN_INFERENCE)
    attack.fabricate_trigger(loader)
    attack.inject_backdoor(loader, test_loader, epochs=VILLAIN_INJECTION)
    cda, asr = attack.evaluate(test_loader)
    total = VILLAIN_WARMUP + VILLAIN_INFERENCE + VILLAIN_INJECTION
    return {"acc": cda, "metric": asr, "lia": 100.0 * attack.label_inference_accuracy(), "attack_epochs": total,
            "iterations": len(loader) * total, "unit": "training steps (train batches x all VILLAIN epochs)",
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


TRAINING_ATTACKS = {"label_leakage": r3elu_label_leakage, "villain": r3elu_villain,
                    "poison_client": r3elu_poison_client}


if __name__ == "__main__":
    if QUICK_TEST:
        EPSILON_VALUES, EPOCHS_PER_EPS = [1.0], 1
        MAX_IMAGES, ITERATIONS, AE_EPOCHS, FSHA_EPOCHS, UNSPLIT_STEPS = 4, 20, 2, 1, 20
        LEAK_EPOCHS = VILLAIN_WARMUP = VILLAIN_INFERENCE = VILLAIN_INJECTION = POISON_TRAIN_EPOCHS = 1
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
        if attack_key not in ATTACKS_TO_RUN:
            continue
        print("\n" + "#" * 78)
        print(f"#   R3eLU vs {attack_key} -- {ATTACK_INFO[attack_key][0]}")
        print(f"#   MODEL: {model_label} | CUT LAYER: {cut_label} | {Config.DATASET}")
        print("#" * 78)
        args = {"label_leakage": (n_features, base_dataset, in_channels),
                "villain": (n_features, base_dataset, test_loader, in_channels),
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

    if "poison_server" in ATTACKS_TO_RUN:
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
            print("    Run defence_runner/run_backdoor_poison.py first to produce it. Skipping.")
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

    order = {a: i for i, a in enumerate(ATTACKS_TO_RUN)}
    all_rows.sort(key=lambda r: order.get(r["attack"], 99))
    for r in all_rows:
        if r["attack_family"] == "reconstruction":
            print_run_info(r)
    print_summary_table(all_rows, f"R3eLU vs ALL ATTACKS -- {model_label} | cut {cut_label} | {Config.DATASET}")
    print_defense_attack_matrix(all_rows)
    save_results(all_rows, all_out)
    print(f"\n  Saved -> {all_out}")
