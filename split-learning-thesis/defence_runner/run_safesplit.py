"""
safesplit_run.py -- SafeSplit vs ALL 8 attacks, same structure as zorro_run.py.

Place in:  defence_runner/safesplit_run.py
Run from the repo root:
    python -m defence_runner.safesplit_run --model PyramidCNN
    python -m defence_runner.safesplit_run --model KAGN
Useful options:
    --attacks VILLAIN,BackdoorPoison_Client      run only these attacks
    --variant pmr20 --pmr 0.2                    20% of turns malicious, rest benign clients
    --variant strict --freq-threshold 2.0 --rot-threshold 1.5
    --server-only-rollback                       old behaviour (client keeps its poisoned part)

How SafeSplit is used here (existing_defense/safesplit_defense.py; Rieger et al., NDSS 2025)
------------------------------------------------------------------------------------------
SafeSplit is a SERVER-side defense. Around every client "turn" the server saves its model,
then scores how far the server model moved (DCT frequency distance and rotation angle,
z-scored against recent benign history). If both look anomalous (or the combined score is
very high) the turn is flagged and the server is rolled back to the pre-turn checkpoint.

Same wiring pattern as zorro_run.py
  * Training: one epoch = one turn of a benign client, SafeSplit in the loop.
  * Attacks 1-5 (WhiteBox, UnSplit, AE, FSHA, LabelLeakage): no hook; run on the SafeSplit-trained model.
  * VILLAIN and Backdoor(Client): the attacker's training is driven one turn (epoch) at a time.
    Before the malicious turns, `--seed-turns` BENIGN turns run through SafeSplit so its z-score
    history is real. `--pmr` < 1 interleaves benign-client turns with the malicious ones.
  * Backdoor(Server): the attacker IS the server that runs SafeSplit -> runs unmonitored.
On a flag the WHOLE model is rolled back (server by SafeSplit, client from a snapshot taken before
the turn), like Zorro's benchmark-model reload. The attacker's NEXT turn after a flag is skipped
(SafeSplit's "flag client, retry next epoch"), tracked per client id.
Per-turn logs go to results/safesplit_turns_<attack>_<tag>_<variant>.csv.
Stats: PDR = % malicious turns flagged, FPR = % benign turns flagged (seed turns excluded).
"""
import os
import sys
import time
import random
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):          # SafeSplit prints emoji; legacy Windows code pages would crash
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from dataset import DatasetLoader
from all_attacks.attack_unsplit import UnSplitAttack
from all_attacks.attacks_whitebox import WhiteBoxInversionAttack, AttackMetricsTracker
from all_attacks.ae_decoder_attack import run_ae_decoder_attack
from all_attacks.fsha_attack import FSHAAttack
from all_attacks.label_leakage_attack import GradientNormLabelLeakageAttack, build_binary_split_loaders
from all_attacks.villain_backdoor_attack import VILLAINBackdoorAttack, build_indexed_loader
from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel
from existing_defense.safesplit_defense import SafeSplitDefense

BLANK = "—"
DEFENSE_NAME = "safesplit"
EXPECTED_ATTACKS = ['WhiteBox', 'UnSplit', 'AE_Decoder', 'FSHA', 'LabelLeakage',
                    'VILLAIN', 'BackdoorPoison_Client', 'BackdoorPoison_Server']
DEFENSE_ACTIVE = {'WhiteBox': 'no (no hook: SafeSplit does not alter smashed data)',
                  'UnSplit': 'no (no hook)', 'AE_Decoder': 'no (no hook)', 'FSHA': 'no (no hook)',
                  'LabelLeakage': 'no (no hook)',
                  'VILLAIN': 'yes (injection phase, one turn per epoch)',
                  'BackdoorPoison_Client': 'yes (one turn per epoch)',
                  'BackdoorPoison_Server': 'no (attacker is the server; outside SafeSplit threat model)'}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_checked(module, state, what):
    missing, unexpected = module.load_state_dict(state, strict=False)
    assert not missing, f"[{what}] checkpoint is missing keys: {list(missing)[:5]}"
    assert not unexpected, f"[{what}] checkpoint has unexpected keys: {list(unexpected)[:5]}"


def materialize_lazy(client, server, device, in_channels=3, image_size=32):
    """The server has nn.LazyLinear; SafeSplit clones weights before epoch 1, so initialise it first."""
    was = (client.training, server.training)
    client.eval(); server.eval()
    with torch.no_grad():
        server(client(torch.zeros(2, in_channels, image_size, image_size, device=device)))
    client.train(was[0]); server.train(was[1])


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def cfg_of(args):
    return {'freq_low_ratio': args.freq_low_ratio, 'freq_threshold': args.freq_threshold,
            'rot_threshold': args.rot_threshold, 'combined_threshold': args.combined_threshold,
            'max_checkpoints': args.max_checkpoints, 'seed_turns': args.seed_turns,
            'rollback_client': not args.server_only_rollback}


def make_defense(args, server_model, device):
    return SafeSplitDefense(server_model=server_model, freq_low_ratio=args.freq_low_ratio,
                            freq_threshold=args.freq_threshold, rot_threshold=args.rot_threshold,
                            combined_threshold=args.combined_threshold, max_checkpoints=args.max_checkpoints,
                            device=str(device))


# ═════════════════════════════════════════════════════════════════════════════
#  SafeSplit guard: one place that runs a turn, logs it, rolls back, and counts
# ═════════════════════════════════════════════════════════════════════════════
class SafeSplitGuard:
    def __init__(self, args, server_model, device):
        self.args = args
        self.d = make_defense(args, server_model, device)
        self.log = []
        self.skipped = 0
        self.prev_flagged = {}            # client id -> was flagged on its previous turn

    def _scores(self):
        h = getattr(self.d, '_detection_history', None)
        if h and isinstance(h[-1], dict):
            r = h[-1]
            return (float(r.get('freq_anomaly_score', float('nan'))), float(r.get('rot_anomaly_score', float('nan'))),
                    float(r.get('combined_score', float('nan'))))
        return float('nan'), float('nan'), float('nan')

    def should_skip(self, cid):
        return bool(self.prev_flagged.get(cid, False))

    def turn(self, turn, cid, client, server, run_epoch, poisoned, phase='run'):
        snap = cpu_state(client)
        self.d.before_client_training(turn, cid, server)
        run_epoch()
        flagged = bool(self.d.after_client_training(turn, cid, server))     # SafeSplit rolls the server back itself
        if flagged and not self.args.server_only_rollback:
            client.load_state_dict(snap)                                    # discard the whole turn
        f, r, cmb = self._scores()
        rec = {'turn': turn, 'client': cid, 'phase': phase, 'poisoned': poisoned, 'flagged': flagged,
               'skipped': False, 'freq_score': f, 'rot_score': r, 'combined': cmb}
        self.log.append(rec)
        self.prev_flagged[cid] = flagged
        return rec

    def skip(self, turn, cid, poisoned):
        self.skipped += 1
        self.prev_flagged[cid] = False
        self.log.append({'turn': turn, 'client': cid, 'phase': 'run', 'poisoned': poisoned, 'flagged': False,
                         'skipped': True, 'freq_score': float('nan'), 'rot_score': float('nan'),
                         'combined': float('nan')})

    def summary(self):
        run = [r for r in self.log if r['phase'] == 'run' and not r['skipped']]
        mal = [r for r in run if r['poisoned']]; ben = [r for r in run if not r['poisoned']]
        mf = sum(r['flagged'] for r in mal); bf = sum(r['flagged'] for r in ben)
        first = next((r['turn'] for r in mal if r['flagged']), None)
        return {'malicious_turns': len(mal), 'benign_turns': len(ben), 'malicious_flagged': mf,
                'benign_flagged': bf, 'PDR': 100.0 * mf / max(1, len(mal)), 'FPR': 100.0 * bf / max(1, len(ben)),
                'rollbacks': mf + bf, 'skipped': self.skipped, 'first_detection_turn': first}


def seed_history(guard, atk, benign_epoch, n):
    """Real benign turns through SafeSplit so its z-score history is not empty when the attacker starts."""
    for i in range(1, n + 1):
        guard.turn(i, f'seed_{i}', atk.client_model, atk.server_model, benign_epoch, poisoned=False, phase='seed')
    print(f"  [SafeSplit] seeded with {n} benign turns "
          f"(flagged during seeding: {sum(r['flagged'] for r in guard.log if r['phase']=='seed')})")


def make_schedule(n_malicious, pmr, seed):
    if pmr >= 1.0:
        return ['M'] * n_malicious
    total = int(round(n_malicious / pmr))
    rng = random.Random(seed)
    pos = set(rng.sample(range(total), n_malicious))
    return ['M' if i in pos else 'B' for i in range(total)]


def guarded_turns(atk, guard, schedule, malicious_epoch, benign_epoch, test_loader, label, turn0):
    hist = {'epoch': [], 'asr': [], 'cda': []}
    for i, kind in enumerate(schedule, 1):
        t = turn0 + i
        cid = 'attacker' if kind == 'M' else f'benign_{i}'
        if kind == 'M' and guard.should_skip(cid):
            guard.skip(t, cid, True)
            msg = "SKIPPED (attacker flagged on its previous turn)"
        else:
            rec = guard.turn(t, cid, atk.client_model, atk.server_model,
                             malicious_epoch if kind == 'M' else benign_epoch, poisoned=(kind == 'M'))
            msg = ("FLAGGED -> rolled back" if rec['flagged'] else "accepted") + \
                  f" | freq {rec['freq_score']:.2f} rot {rec['rot_score']:.2f}"
        cda, asr = atk.evaluate(test_loader)
        hist['epoch'].append(i); hist['asr'].append(asr); hist['cda'].append(cda)
        print(f"  [SafeSplit] {label} turn {i:2d}/{len(schedule)} ({kind}) {msg} | ASR {asr:.2f}% | CDA {cda:.2f}%")
    atk.history = hist
    return guard.summary()


# ═════════════════════════════════════════════════════════════════════════════
#  SafeSplit-monitored split-learning trainer (benign client, one epoch = one turn)
# ═════════════════════════════════════════════════════════════════════════════
class SafeSplitTrainer:
    CLIENT_ID = 0

    def __init__(self, client, server, train_loader, test_loader, device, ckpt_path, args):
        self.client, self.server, self.device, self.ckpt_path = client, server, device, ckpt_path
        self.train_loader, self.test_loader = train_loader, test_loader
        self.args = args
        self.client_opt = torch.optim.Adam(client.parameters(), lr=Config.LEARNING_RATE)
        self.server_opt = torch.optim.Adam(server.parameters(), lr=Config.LEARNING_RATE)
        self.client_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.client_opt, patience=5, factor=0.5)
        self.server_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.server_opt, patience=5, factor=0.5)
        self.criterion = nn.CrossEntropyLoss()
        self.defense = make_defense(args, server, device)
        self.losses, self.train_accs, self.test_accs, self.skipped = [], [], [], 0

    def _epoch(self):
        self.client.train(); self.server.train()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(self.train_loader, desc="  SafeSplit benign turn", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            self.client_opt.zero_grad()
            z = self.client(x)
            z_leaf = z.detach().requires_grad_(True)
            self.server_opt.zero_grad()
            out = self.server(z_leaf)
            loss = self.criterion(out, y)
            loss.backward(); self.server_opt.step()
            z.backward(z_leaf.grad); self.client_opt.step()
            run_loss += loss.item(); total += y.size(0); correct += out.argmax(1).eq(y).sum().item()
        return run_loss / len(self.train_loader), 100.0 * correct / total

    @torch.no_grad()
    def evaluate(self):
        self.client.eval(); self.server.eval()
        c = t = 0
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            c += self.server(self.client(x)).argmax(1).eq(y).sum().item(); t += y.size(0)
        return 100.0 * c / t

    def train(self):
        print("\n" + "=" * 60 + f"\n   SAFESPLIT SPLIT LEARNING — {Config.MODEL_NAME} | cut {Config.CUT_LAYER}\n" + "=" * 60)
        best = 0.0
        for epoch in range(1, Config.EPOCHS + 1):
            if self.defense.should_skip_client(self.CLIENT_ID, epoch):
                self.skipped += 1
                print(f"  Epoch {epoch:3d}/{Config.EPOCHS} | SKIPPED (client flagged last turn)")
                continue
            snap = cpu_state(self.client)
            self.defense.before_client_training(epoch, self.CLIENT_ID, self.server)
            loss, tr = self._epoch()
            flagged = bool(self.defense.after_client_training(epoch, self.CLIENT_ID, self.server))
            if flagged:
                self.client.load_state_dict(snap)
            te = self.evaluate()
            self.losses.append(loss); self.train_accs.append(tr); self.test_accs.append(te)
            self.client_sched.step(te); self.server_sched.step(te)
            st = self.defense.get_stats()
            print(f"  Epoch {epoch:3d}/{Config.EPOCHS} | Loss {loss:.4f} | Train {tr:.2f}% | Test {te:.2f}% | "
                  f"SafeSplit flagged {st['flagged_as_poisoned']}/{st['total_clients_analyzed']}"
                  f"{' (rolled back)' if flagged else ''}")
            if te > best:
                best = te
                torch.save({'epoch': epoch, 'client_state': self.client.state_dict(),
                            'server_state': self.server.state_dict(), 'best_acc': best,
                            'dataset': Config.DATASET, 'model_type': Config.MODEL_NAME,
                            'cut_layer': Config.CUT_LAYER, 'defense': DEFENSE_NAME,
                            'safesplit_cfg': cfg_of(self.args)}, self.ckpt_path)
        self.defense.print_stats()
        print(f"  SafeSplit benign training done. Best test accuracy {best:.2f}% | skipped turns {self.skipped}")
        return self.losses, self.train_accs, self.test_accs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=Config.MODEL_NAME, choices=['KAGN', 'PyramidCNN', 'Vanilla_SL'])
    ap.add_argument('--cut', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip-attacks', action='store_true')
    ap.add_argument('--attacks', default=','.join(EXPECTED_ATTACKS),
                    help="comma-separated subset of: " + ','.join(EXPECTED_ATTACKS))
    ap.add_argument('--variant', default='full', help="name used in output files (ablation id)")
    ap.add_argument('--freq-low-ratio', type=float, default=0.1)
    ap.add_argument('--freq-threshold', type=float, default=2.5)
    ap.add_argument('--rot-threshold', type=float, default=2.0)
    ap.add_argument('--combined-threshold', type=float, default=1.5)
    ap.add_argument('--max-checkpoints', type=int, default=10)
    ap.add_argument('--seed-turns', type=int, default=5, help="benign turns run through SafeSplit before the attacker starts")
    ap.add_argument('--server-only-rollback', action='store_true',
                    help="on a flag roll back only the server (attacker keeps its client part)")
    ap.add_argument('--pmr', type=float, default=1.0, help="fraction of turns that are malicious")
    args = ap.parse_args()

    Config.MODEL_NAME, Config.CUT_LAYER, Config.DATASET = args.model, args.cut, 'CIFAR10'
    set_seed(args.seed)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    TAG = f"{Config.MODEL_NAME.lower()}_cut{Config.CUT_LAYER}_{Config.DATASET}"
    VTAG = f"{TAG}_{args.variant}"
    os.makedirs(Config.SAVE_DIR, exist_ok=True); os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    selected = [a.strip() for a in args.attacks.split(',') if a.strip()]
    assert set(selected) <= set(EXPECTED_ATTACKS), f"unknown attack(s): {set(selected) - set(EXPECTED_ATTACKS)}"
    print(f"Device: {device} | Defense: SAFESPLIT ({args.variant}) | {cfg_of(args)} | pmr {args.pmr}")

    train_loader, test_loader = DatasetLoader(dataset_name=Config.DATASET).get_loaders()
    in_channels = 3
    checkpoint_path = f"{Config.SAVE_DIR}/best_{DEFENSE_NAME}_{TAG}.pth"

    def build_client():
        if Config.MODEL_NAME == "KAGN":
            return KAGNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels, degree=Config.DEGREE).to(device)
        if Config.MODEL_NAME == "PyramidCNN":
            return PyramidCNNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels).to(device)
        return ClientModel(in_channels=in_channels).to(device)

    def build_server(num_classes):
        if Config.MODEL_NAME == "KAGN":
            return KAGNServerModel(cut_layer=Config.CUT_LAYER, num_classes=num_classes,
                                   in_channels=in_channels, degree=Config.DEGREE).to(device)
        if Config.MODEL_NAME == "PyramidCNN":
            return PyramidCNNServerModel(cut_layer=Config.CUT_LAYER, num_classes=num_classes,
                                         in_channels=in_channels).to(device)
        return ServerModel(num_classes=num_classes).to(device)

    def _load_ckpt():
        ck = torch.load(checkpoint_path, map_location=device)
        assert ck.get('defense') == DEFENSE_NAME, "Checkpoint is not a SafeSplit checkpoint"
        if Config.MODEL_NAME != "Vanilla_SL":
            assert ck['cut_layer'] == Config.CUT_LAYER
        return ck

    def build_fresh_split(num_classes):
        c, s = build_client(), build_server(num_classes)
        ck = _load_ckpt()
        load_checked(c, ck['client_state'], 'safesplit client')
        if num_classes == Config.NUM_CLASSES:
            load_checked(s, ck['server_state'], 'safesplit server')
        else:
            materialize_lazy(c, s, device, in_channels)
        return c, s

    client_model, server_model = build_client(), build_server(Config.NUM_CLASSES)
    materialize_lazy(client_model, server_model, device, in_channels)

    if os.path.exists(checkpoint_path):
        print(f"\n[✓] Found SafeSplit checkpoint: {checkpoint_path} — skipping training")
    else:
        print(f"\n[!] No checkpoint at {checkpoint_path} — training with SafeSplit (default settings)")
        default_args = ap.parse_args([])
        trainer = SafeSplitTrainer(client_model, server_model, train_loader, test_loader, device,
                                   checkpoint_path, default_args)
        losses, tr_accs, te_accs = trainer.train()
        pd.DataFrame({'epoch': range(1, len(losses) + 1), 'train_loss': losses, 'train_accuracy': tr_accs,
                      'test_accuracy': te_accs}).to_csv(f"{Config.RESULTS_DIR}/{DEFENSE_NAME}_results_{TAG}.csv",
                                                        index=False)
        pd.DataFrame([{**trainer.defense.get_stats(), 'skipped_turns': trainer.skipped,
                       **{f'cfg_{k}': v for k, v in cfg_of(default_args).items()}}]).to_csv(
            f"{Config.RESULTS_DIR}/{DEFENSE_NAME}_stats_{TAG}.csv", index=False)
        pd.DataFrame(trainer.defense._detection_history).to_csv(
            f"{Config.RESULTS_DIR}/{DEFENSE_NAME}_turns_train_{TAG}.csv", index=False)

    ck = _load_ckpt()
    load_checked(client_model, ck['client_state'], 'safesplit client')
    load_checked(server_model, ck['server_state'], 'safesplit server')
    client_model.eval(); server_model.eval()
    c = t = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            c += server_model(client_model(x)).argmax(1).eq(y).sum().item(); t += y.size(0)
    clean_acc = 100.0 * c / t
    print(f"\n  Clean test accuracy (SafeSplit, {Config.MODEL_NAME}, cut {Config.CUT_LAYER}): {clean_acc:.2f}%")
    if args.skip_attacks:
        sys.exit(0)

    # ── attack parameters: identical to zorro_run.py ──────────────────────
    MAX_IMAGES, ITERATIONS = 32, 1000
    HIJACK_EPOCHS, CRITIC_ITERS = 5, 5
    LEAKAGE_EPOCHS, POSITIVE_CLASS, POSITIVE_RATIO, LEAKAGE_BATCH, LEAKAGE_LR = 5, 0, 0.1, 128, 1e-4
    WARMUP_EPOCHS, INFERENCE_EPOCHS, INJECTION_EPOCHS, VILLAIN_BATCH = 5, 5, 10, 128
    TARGET_LABEL, TRIGGER_BETA, TRIGGER_FRACTION = 0, 1.0, 0.5
    DROPOUT_KEEP, GAMMA_LOW, GAMMA_HIGH, POISON_RATE, CANDIDATES = 0.75, 0.6, 1.2, 0.01, 14
    BD_TARGET, BD_RATE, BD_PATCH, BD_TRIG, BD_EPOCHS, BD_SURR_EPOCHS = 0, 0.05, 4, 1.0, 10, 5

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)
    results, failures, activity = {}, {}, {}
    R = Config.RESULTS_DIR

    def run_attack(name, title, fn):
        if name not in selected:
            return
        print("\n" + "=" * 60 + f"\n  {title} vs SAFESPLIT[{args.variant}] — {Config.MODEL_NAME} (cut {Config.CUT_LAYER})\n" + "=" * 60)
        t0 = time.time()
        try:
            results[name] = fn()
            print(f"  [done in {time.time()-t0:.0f}s]")
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
            print(f"  [FAILED] {failures[name]}")
            traceback.print_exc()

    def record_activity(name, guard, stats):
        activity[name] = stats
        pd.DataFrame(guard.log).to_csv(f"{R}/{DEFENSE_NAME}_turns_{name.lower()}_{VTAG}.csv", index=False)
        print(f"  [SafeSplit] {name}: PDR {stats['PDR']:.1f}% ({stats['malicious_flagged']}/{stats['malicious_turns']}) | "
              f"FPR {stats['FPR']:.1f}% ({stats['benign_flagged']}/{stats['benign_turns']}) | "
              f"rollbacks {stats['rollbacks']} | skipped {stats['skipped']}")

    def attack_whitebox():
        atk = WhiteBoxInversionAttack(client_model=client_model, dataset=Config.DATASET, iterations=ITERATIONS, lr=1e-2)
        tracker, done = AttackMetricsTracker(), 0
        for inputs, _ in test_loader:
            if done >= MAX_IMAGES:
                break
            inputs = inputs.to(device)[:MAX_IMAGES - done]
            with torch.no_grad():
                smashed = client_model(inputs)
            rec = atk.reconstruct(smashed, inputs.shape)
            tracker.log_batch(torch.clamp(inputs * std + mean, 0, 1), rec)
            done += inputs.shape[0]
        s = tracker.get_summary()
        pd.DataFrame([s]).to_csv(f"{R}/{DEFENSE_NAME}_attack_whitebox_{VTAG}.csv", index=False)
        print(f"  PSNR {s['mean_psnr']:.2f} dB | SSIM {s['mean_ssim']:.4f}")
        return s

    def attack_unsplit():
        atk = UnSplitAttack(client_model=client_model, in_channels=in_channels, clone_builder=build_client,
                            main_iters=Config.unsplit_main_iters, input_iters=Config.unsplit_input_iters,
                            model_iters=Config.unsplit_model_iters)
        s = atk.run_attack(test_loader, num_batches=MAX_IMAGES // 32 or 1)
        pd.DataFrame([s]).to_csv(f"{R}/{DEFENSE_NAME}_attack_unsplit_{VTAG}.csv", index=False)
        print(f"  PSNR {s['psnr']:.2f} dB | SSIM {s['ssim']:.4f}")
        return s

    def attack_ae():
        s, o, r = run_ae_decoder_attack(client_model=client_model, train_loader=train_loader, test_loader=test_loader,
                                        device=device, dataset=Config.DATASET, preprocess_fn=None, ae_epochs=50,
                                        label=f'{Config.MODEL_NAME} SafeSplit AE Decoder')
        pd.DataFrame([s]).to_csv(f"{R}/{DEFENSE_NAME}_attack_ae_decoder_{VTAG}.csv", index=False)
        return s

    def attack_fsha():
        atk = FSHAAttack(client_model=client_model, in_channels=in_channels, dataset=Config.DATASET,
                         pilot_builder=build_client, critic_iters=CRITIC_ITERS)
        atk.hijack(train_loader, test_loader, epochs=HIJACK_EPOCHS)
        s = atk.reconstruct(train_loader, num_images=MAX_IMAGES)
        pd.DataFrame([s]).to_csv(f"{R}/{DEFENSE_NAME}_attack_fsha_{VTAG}.csv", index=False)
        return s

    def attack_label_leakage():
        btr, bte = build_binary_split_loaders(train_loader.dataset, target_class=POSITIVE_CLASS,
                                              positive_ratio=POSITIVE_RATIO, batch_size=LEAKAGE_BATCH)
        lc, ls = build_fresh_split(num_classes=1)
        atk = GradientNormLabelLeakageAttack(client_model=lc, server_model=ls, dataset=Config.DATASET,
                                             target_class=POSITIVE_CLASS, learning_rate=LEAKAGE_LR)
        s = atk.run(btr, bte, epochs=LEAKAGE_EPOCHS)
        pd.DataFrame([s]).to_csv(f"{R}/{DEFENSE_NAME}_attack_label_leakage_{VTAG}.csv", index=False)
        return s

    def attack_villain():
        idx_loader = build_indexed_loader(train_loader.dataset, batch_size=VILLAIN_BATCH, shuffle=True)
        vc, vs = build_fresh_split(num_classes=Config.NUM_CLASSES)
        atk = VILLAINBackdoorAttack(client_model=vc, server_model=vs, base_dataset=train_loader.dataset,
                                    dataset=Config.DATASET, num_classes=Config.NUM_CLASSES, target_label=TARGET_LABEL,
                                    beta=TRIGGER_BETA, trigger_fraction=TRIGGER_FRACTION, dropout_keep=DROPOUT_KEEP,
                                    gamma_low=GAMMA_LOW, gamma_high=GAMMA_HIGH, poison_rate=POISON_RATE,
                                    candidates_per_batch=CANDIDATES)
        atk.warmup(idx_loader, epochs=WARMUP_EPOCHS)
        base, _ = atk.evaluate(test_loader)
        atk.infer_labels(idx_loader, epochs=INFERENCE_EPOCHS)
        atk.fabricate_trigger(idx_loader)

        def benign_epoch():
            atk.client_model.train(); atk.server_model.train()
            for x, y, _ in tqdm(idx_loader, desc="  benign turn", leave=False):
                atk._split_step(x.to(atk.device), y.to(atk.device))

        guard = SafeSplitGuard(args, atk.server_model, device)
        atk._set_client_learning_rate(atk.attack_learning_rate)
        seed_history(guard, atk, benign_epoch, args.seed_turns)
        stats = guarded_turns(atk, guard, make_schedule(INJECTION_EPOCHS, args.pmr, args.seed),
                              malicious_epoch=lambda: atk.inject_backdoor(idx_loader, test_loader, epochs=1),
                              benign_epoch=benign_epoch, test_loader=test_loader, label='VILLAIN',
                              turn0=args.seed_turns)
        record_activity('VILLAIN', guard, stats)
        s = atk.summarise(test_loader=test_loader, clean_baseline=base)
        pd.DataFrame([{**s, **{f'safesplit_{k}': v for k, v in stats.items()}}]).to_csv(
            f"{R}/{DEFENSE_NAME}_attack_villain_{VTAG}.csv", index=False)
        return s

    def attack_backdoor(mode):
        def _run():
            bc, bs = build_fresh_split(num_classes=Config.NUM_CLASSES)
            kw = dict(client_model=bc, server_model=bs, base_dataset=train_loader.dataset, dataset=Config.DATASET,
                      num_classes=Config.NUM_CLASSES, mode=mode, target_label=BD_TARGET, poison_rate=BD_RATE,
                      patch_size=BD_PATCH, trigger_value=BD_TRIG,
                      model_tag=f"{DEFENSE_NAME}_{args.variant}_{Config.MODEL_NAME.lower()}_sl")
            if mode == 'server':
                kw['surrogate_builder'] = build_client
            atk = BackdoorPoisonAttack(**kw)
            extra = {}
            if mode == 'server':
                atk.pretrain_server_backdoor(test_loader, epochs=BD_SURR_EPOCHS)
                print("  [SafeSplit] NOT applicable: the attacker is the server. Running unmonitored.")
                atk.train(train_loader, test_loader, epochs=BD_EPOCHS)
            else:
                def benign_epoch():
                    atk.client_model.train(); atk.server_model.train()
                    for x, y in tqdm(train_loader, desc="  benign turn", leave=False):
                        atk._split_step(x.to(atk.device), y.to(atk.device))

                def malicious_epoch():
                    atk.client_model.train(); atk.server_model.train()
                    for x, y in tqdm(train_loader, desc="  poisoned turn", leave=False):
                        x, y, _ = atk._poison_batch(x.to(atk.device), y.to(atk.device))
                        atk._split_step(x, y)

                guard = SafeSplitGuard(args, atk.server_model, device)
                seed_history(guard, atk, benign_epoch, args.seed_turns)
                stats = guarded_turns(atk, guard, make_schedule(BD_EPOCHS, args.pmr, args.seed),
                                      malicious_epoch=malicious_epoch, benign_epoch=benign_epoch,
                                      test_loader=test_loader, label='Backdoor(Client)', turn0=args.seed_turns)
                record_activity('BackdoorPoison_Client', guard, stats)
                extra = {f'safesplit_{k}': v for k, v in stats.items()}
            s = atk.summarise()
            pd.DataFrame([{**s, **extra}]).to_csv(f"{R}/{DEFENSE_NAME}_attack_backdoor_poison_{mode}_{VTAG}.csv",
                                                  index=False)
            return s
        return _run

    run_attack('WhiteBox', 'WHITE-BOX ATTACK', attack_whitebox)
    run_attack('UnSplit', 'UNSPLIT ATTACK', attack_unsplit)
    run_attack('AE_Decoder', 'AE DECODER ATTACK', attack_ae)
    run_attack('FSHA', 'FSHA ATTACK', attack_fsha)
    run_attack('LabelLeakage', 'LABEL LEAKAGE ATTACK', attack_label_leakage)
    run_attack('VILLAIN', 'VILLAIN BACKDOOR ATTACK', attack_villain)
    run_attack('BackdoorPoison_Client', 'BACKDOOR POISONING (CLIENT)', attack_backdoor('client'))
    run_attack('BackdoorPoison_Server', 'BACKDOOR POISONING (SERVER)', attack_backdoor('server'))

    # ── summary table (same layout as zorro_run.py) ───────────────────────
    METRICS = {'WhiteBox': {'psnr': 'mean_psnr', 'ssim': 'mean_ssim'}, 'UnSplit': {'psnr': 'psnr', 'ssim': 'ssim'},
               'AE_Decoder': {'psnr': 'mean_psnr', 'ssim': 'mean_ssim'},
               'FSHA': {'psnr': 'psnr', 'ssim': 'ssim', 'mse': 'mse'},
               'LabelLeakage': {'auc': 'q95_norm_leak_auc_cut'},
               'VILLAIN': {'lia': 'lia', 'asr': 'asr', 'cda': 'cda'},
               'BackdoorPoison_Client': {'asr': 'asr', 'cda': 'cda'}, 'BackdoorPoison_Server': {'asr': 'asr', 'cda': 'cda'}}
    COLS = [('psnr', 'PSNR', 2), ('ssim', 'SSIM', 4), ('mse', 'MSE', 5), ('auc', 'LeakAUC', 4),
            ('lia', 'LIA%', 2), ('asr', 'ASR%', 2), ('cda', 'CDA%', 2)]

    def cell(a, col, dec):
        if col not in METRICS[a] or a not in selected:
            return BLANK
        if a in failures:
            return "ERR"
        if a not in results:
            return BLANK
        v = results[a].get(METRICS[a][col])
        return "MISSING" if v is None else f"{float(v):.{dec}f}"

    W = 100
    print("\n" + "=" * W + f"\n    SAFESPLIT[{args.variant}] — {Config.MODEL_NAME} | cut {Config.CUT_LAYER} | "
          f"clean acc {clean_acc:.2f}%\n" + "=" * W)
    print(f"{'Attack':<24}" + "".join(f"{h:>10}" for _, h, _ in COLS))
    for a in EXPECTED_ATTACKS:
        print(f"  {a:<22}" + "".join(f"{cell(a, c_, d):>10}" for c_, _, d in COLS))
    for name, st in activity.items():
        print(f"  SAFESPLIT activity {name}: PDR {st['PDR']:.1f}% | FPR {st['FPR']:.1f}% | "
              f"rollbacks {st['rollbacks']} | skipped {st['skipped']} | first detection turn {st['first_detection_turn']}")
    for a, msg in failures.items():
        print(f"  FAILED  {a}: {msg}")

    rows = []
    for a in selected:
        row = {'defense': DEFENSE_NAME, 'variant': args.variant, 'model': Config.MODEL_NAME,
               'cut_layer': Config.CUT_LAYER, 'dataset': Config.DATASET, 'seed': args.seed, 'clean_acc': clean_acc,
               'attack': a, 'safesplit_in_loop': DEFENSE_ACTIVE[a], 'pmr': args.pmr,
               'status': 'FAILED' if a in failures else 'ok', 'error': failures.get(a),
               **{f'cfg_{k}': v for k, v in cfg_of(args).items()}}
        for c_, _, _ in COLS:
            row[c_] = None if (c_ not in METRICS[a] or a in failures or a not in results) else results[a].get(METRICS[a][c_])
        if a in activity:
            row.update({f'safesplit_{k}': v for k, v in activity[a].items()})
        rows.append(row)
    out = f"{R}/{DEFENSE_NAME}_all_attacks_{VTAG}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Saved → {out}")