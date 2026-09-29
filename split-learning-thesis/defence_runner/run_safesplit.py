
import os
import sys
import time
import random
import argparse
import traceback
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SafeSplitDefense prints emoji; legacy Windows code pages would raise UnicodeEncodeError.
for _s in (sys.stdout, sys.stderr):
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
from all_attacks.ae_decoder_attack import run_ae_decoder_attack, save_ae_attack_visualization
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
CLIENT_ID = 0            # single-client setting

# ── SafeSplit hyper-parameters = the defaults of SafeSplitDefense ───────────
SS_FREQ_LOW_RATIO     = 0.1
SS_FREQ_THRESHOLD     = 2.5
SS_ROT_THRESHOLD      = 2.0
SS_COMBINED_THRESHOLD = 1.5
SS_MAX_CHECKPOINTS    = 10

EXPECTED_ATTACKS = ['WhiteBox', 'UnSplit', 'AE_Decoder', 'FSHA', 'LabelLeakage',
                    'VILLAIN', 'BackdoorPoison_Client', 'BackdoorPoison_Server']

# Was SafeSplit actually in the loop for this attack?
DEFENSE_ACTIVE = {'WhiteBox': 'no (no hook)', 'UnSplit': 'no (no hook)', 'AE_Decoder': 'no (no hook)',
                  'FSHA': 'no (no hook)', 'LabelLeakage': 'no (no hook)',
                  'VILLAIN': 'yes (injection phase)', 'BackdoorPoison_Client': 'yes',
                  'BackdoorPoison_Server': 'no (attacker is the server)'}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checked(module, state, what):
    """strict=False silently leaves layers untrained on a mismatch -> fail loudly instead."""
    missing, unexpected = module.load_state_dict(state, strict=False)
    assert not missing, f"[{what}] checkpoint is missing keys (untrained layers): {list(missing)[:5]}"
    assert not unexpected, f"[{what}] checkpoint has unexpected keys: {list(unexpected)[:5]}"


def materialize_lazy(client, server, device, in_channels=3, image_size=32):
    """The server contains nn.LazyLinear, uninitialised until its first forward pass.
    SafeSplit clones/deep-copies the server weights BEFORE the first epoch, which fails on
    uninitialised parameters, so run one dry forward pass (eval mode: no BN/dropout side effects)."""
    was = (client.training, server.training)
    client.eval(); server.eval()
    with torch.no_grad():
        server(client(torch.zeros(2, in_channels, image_size, image_size, device=device)))
    client.train(was[0]); server.train(was[1])


# ═════════════════════════════════════════════════════════════════════════════
#  SafeSplit helpers
# ═════════════════════════════════════════════════════════════════════════════
def make_defense(server_model, device, benign_hist=None):
    """Fresh SafeSplitDefense; optionally seeded with the benign z-score history
    recorded during clean training (so a malicious turn is judged against a benign past)."""
    d = SafeSplitDefense(server_model=server_model,
                         freq_low_ratio=SS_FREQ_LOW_RATIO,
                         freq_threshold=SS_FREQ_THRESHOLD,
                         rot_threshold=SS_ROT_THRESHOLD,
                         combined_threshold=SS_COMBINED_THRESHOLD,
                         max_checkpoints=SS_MAX_CHECKPOINTS,
                         device=str(device))
    if benign_hist:
        d.freq_analyzer._distance_history = [float(v) for v in benign_hist.get('freq', [])]
        d.rot_analyzer._rotation_history = [float(v) for v in benign_hist.get('rot', [])]
    return d


def export_hist(defense):
    return {'freq': [float(v) for v in defense.freq_analyzer._distance_history],
            'rot': [float(v) for v in defense.rot_analyzer._rotation_history]}


def guarded_epochs(atk, defense, n_epochs, run_epoch, test_loader, label):
    """
    Drive an attack's training ONE EPOCH AT A TIME with SafeSplit around each epoch.
    `atk` must expose .server_model, .history {'epoch','asr','cda'} and .evaluate(test_loader)
    -> (clean_acc, asr). `run_epoch()` must train exactly one epoch and append one history entry.
    """
    stats = {'accepted': 0, 'rolled_back': 0, 'skipped': 0, 'flagged_epochs': []}
    for e in range(1, n_epochs + 1):
        if defense.should_skip_client(CLIENT_ID, e):
            stats['skipped'] += 1
            cda, asr = atk.evaluate(test_loader)
            atk.history['epoch'].append(e)
            atk.history['asr'].append(asr)
            atk.history['cda'].append(cda)
            print(f"  [SafeSplit] {label} turn {e:2d}/{n_epochs}: SKIPPED (flagged last turn) | "
                  f"ASR {asr:.2f}% | CDA {cda:.2f}%")
            continue

        defense.before_client_training(e, CLIENT_ID, atk.server_model)
        run_epoch()
        flagged = defense.after_client_training(e, CLIENT_ID, atk.server_model)

        if flagged:
            stats['rolled_back'] += 1
            stats['flagged_epochs'].append(e)
            # the attack logged ASR/CDA of the poisoned model; replace with post-rollback values
            cda, asr = atk.evaluate(test_loader)
            atk.history['asr'][-1] = asr
            atk.history['cda'][-1] = cda
            print(f"  [SafeSplit] {label} turn {e:2d}/{n_epochs}: FLAGGED -> server rolled back | "
                  f"ASR {asr:.2f}% | CDA {cda:.2f}%")
        else:
            stats['accepted'] += 1
            print(f"  [SafeSplit] {label} turn {e:2d}/{n_epochs}: accepted | "
                  f"ASR {atk.history['asr'][-1]:.2f}% | CDA {atk.history['cda'][-1]:.2f}%")

    atk.history['epoch'] = list(range(1, len(atk.history['asr']) + 1))
    return stats


# ═════════════════════════════════════════════════════════════════════════════
#  SafeSplit-monitored split-learning trainer (benign client)
# ═════════════════════════════════════════════════════════════════════════════
class SafeSplitTrainer:
    def __init__(self, client, server, train_loader, test_loader, device, ckpt_path):
        self.client, self.server = client, server
        self.train_loader, self.test_loader = train_loader, test_loader
        self.device, self.ckpt_path = device, ckpt_path

        self.client_opt = torch.optim.Adam(client.parameters(), lr=Config.LEARNING_RATE)
        self.server_opt = torch.optim.Adam(server.parameters(), lr=Config.LEARNING_RATE)
        self.client_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.client_opt, patience=5, factor=0.5)
        self.server_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.server_opt, patience=5, factor=0.5)
        self.criterion = nn.CrossEntropyLoss()
        self.defense = make_defense(server, device)

        self.losses, self.train_accs, self.test_accs = [], [], []
        self.skipped = 0

    def _train_one_epoch(self, epoch_idx):
        self.client.train(); self.server.train()
        run_loss, correct, total = 0.0, 0, 0
        bar = tqdm(self.train_loader, desc=f"  SafeSplit Epoch [{epoch_idx+1}/{Config.EPOCHS}]", leave=False)
        for x, y in bar:
            x, y = x.to(self.device), y.to(self.device)
            self.client_opt.zero_grad()
            z = self.client(x)
            z_leaf = z.detach().requires_grad_(True)

            self.server_opt.zero_grad()
            out = self.server(z_leaf)
            loss = self.criterion(out, y)
            loss.backward()
            self.server_opt.step()

            z.backward(z_leaf.grad)
            self.client_opt.step()

            run_loss += loss.item(); total += y.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            bar.set_postfix(Loss=f"{loss.item():.4f}", Acc=f"{100.*correct/total:.2f}%")
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
        print("\n" + "=" * 60)
        print(f"   SAFESPLIT SPLIT LEARNING — {Config.MODEL_NAME} | cut {Config.CUT_LAYER}")
        print("=" * 60)
        best = 0.0
        for epoch in range(1, Config.EPOCHS + 1):          # turn ids are 1-based
            if self.defense.should_skip_client(CLIENT_ID, epoch):
                self.skipped += 1
                print(f"  Epoch {epoch:3d}/{Config.EPOCHS} | SKIPPED (client flagged last turn)")
                continue

            client_snapshot = deepcopy(self.client.state_dict())
            self.defense.before_client_training(epoch, CLIENT_ID, self.server)
            loss, tr = self._train_one_epoch(epoch - 1)
            flagged = self.defense.after_client_training(epoch, CLIENT_ID, self.server)
            if flagged:
                # SafeSplit restored the server; discard the client's part of the turn too
                self.client.load_state_dict(client_snapshot)

            te = self.evaluate()
            self.losses.append(loss); self.train_accs.append(tr); self.test_accs.append(te)
            self.client_sched.step(te); self.server_sched.step(te)
            st = self.defense.get_stats()
            print(f"  Epoch {epoch:3d}/{Config.EPOCHS} | Loss {loss:.4f} | Train {tr:.2f}% | Test {te:.2f}% | "
                  f"SafeSplit flagged {st['flagged_as_poisoned']}/{st['total_clients_analyzed']}"
                  f"{' (rolled back)' if flagged else ''}")

            if te > best:
                best = te
                torch.save({'epoch': epoch,
                            'client_state': self.client.state_dict(),
                            'server_state': self.server.state_dict(),
                            'best_acc': best,
                            'dataset': Config.DATASET,
                            'model_type': Config.MODEL_NAME,
                            'cut_layer': Config.CUT_LAYER,
                            'defense': DEFENSE_NAME,
                            'safesplit_hist': export_hist(self.defense)}, self.ckpt_path)
        self.defense.print_stats()
        print(f"  SafeSplit training complete. Best test accuracy: {best:.2f}% | skipped turns: {self.skipped}")
        return self.losses, self.train_accs, self.test_accs

    def stats(self):
        s = dict(self.defense.get_stats())
        s['skipped_turns'] = self.skipped
        s.update({'freq_low_ratio': SS_FREQ_LOW_RATIO, 'freq_threshold': SS_FREQ_THRESHOLD,
                  'rot_threshold': SS_ROT_THRESHOLD, 'combined_threshold': SS_COMBINED_THRESHOLD,
                  'max_checkpoints': SS_MAX_CHECKPOINTS})
        return s


def plot_training(losses, tr, te, tag):
    ep = range(1, len(losses) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(ep, losses, 'b-', lw=2); a1.set_title('Training Loss'); a1.grid(alpha=.3)
    a2.plot(ep, tr, 'b-', lw=2, label='Train'); a2.plot(ep, te, 'r-', lw=2, label='Test')
    a2.set_title('Accuracy (%)'); a2.legend(); a2.grid(alpha=.3)
    plt.suptitle(f'SafeSplit {Config.MODEL_NAME} Split Learning — {Config.DATASET} (cut {Config.CUT_LAYER})')
    plt.tight_layout()
    path = f"{Config.RESULTS_DIR}/{DEFENSE_NAME}_training_curves_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Training curves saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=Config.MODEL_NAME, choices=['KAGN', 'PyramidCNN', 'Vanilla_SL'])
    ap.add_argument('--cut', type=int, default=2, help="cut layer (KAGN / PyramidCNN); task requires 2")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip-attacks', action='store_true')
    args = ap.parse_args()

    Config.MODEL_NAME = args.model
    Config.CUT_LAYER = args.cut
    Config.DATASET = 'CIFAR10'
    set_seed(args.seed)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    TAG = f"{Config.MODEL_NAME.lower()}_cut{Config.CUT_LAYER}_{Config.DATASET}"
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    print(f"Device: {device} | Defense: SAFESPLIT | Model: {Config.MODEL_NAME} | "
          f"Cut layer: {Config.CUT_LAYER} | Dataset: {Config.DATASET} | Seed: {args.seed}")

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
        if 'cut_layer' in ck and Config.MODEL_NAME != "Vanilla_SL":
            assert ck['cut_layer'] == Config.CUT_LAYER, \
                f"Checkpoint cut layer {ck['cut_layer']} != requested {Config.CUT_LAYER}"
        assert ck.get('defense', DEFENSE_NAME) == DEFENSE_NAME, "Checkpoint is not a SafeSplit checkpoint"
        return ck

    def build_fresh_client():
        """Untrained client (attacker-side clone for UnSplit / FSHA / server-backdoor surrogate)."""
        return build_client()

    def build_fresh_split(num_classes):
        """Client initialised from the SafeSplit-trained checkpoint; server only if 10-class."""
        c, s = build_client(), build_server(num_classes)
        ck = _load_ckpt()
        load_checked(c, ck['client_state'], 'safesplit client')
        if num_classes == Config.NUM_CLASSES:
            load_checked(s, ck['server_state'], 'safesplit server')
        return c, s

    client_model, server_model = build_client(), build_server(Config.NUM_CLASSES)
    materialize_lazy(client_model, server_model, device, in_channels)

    train_stats = None
    if os.path.exists(checkpoint_path):
        print(f"\n[✓] Found SafeSplit checkpoint: {checkpoint_path} — skipping training")
    else:
        print(f"\n[!] No checkpoint at {checkpoint_path} — training with SafeSplit")
        trainer = SafeSplitTrainer(client_model, server_model, train_loader, test_loader, device, checkpoint_path)
        losses, tr_accs, te_accs = trainer.train()
        plot_training(losses, tr_accs, te_accs, TAG)
        pd.DataFrame({'epoch': range(1, len(losses) + 1), 'train_loss': losses,
                      'train_accuracy': tr_accs, 'test_accuracy': te_accs}
                     ).to_csv(f"{Config.RESULTS_DIR}/{DEFENSE_NAME}_results_{TAG}.csv", index=False)
        train_stats = trainer.stats()
        pd.DataFrame([train_stats]).to_csv(f"{Config.RESULTS_DIR}/{DEFENSE_NAME}_stats_{TAG}.csv", index=False)
        pd.DataFrame(trainer.defense._detection_history).to_csv(
            f"{Config.RESULTS_DIR}/{DEFENSE_NAME}_detections_train_{TAG}.csv", index=False)

    # always evaluate/attack the BEST checkpoint (the one saved by the trainer)
    ck = _load_ckpt()
    load_checked(client_model, ck['client_state'], 'safesplit client')
    load_checked(server_model, ck['server_state'], 'safesplit server')
    benign_hist = ck.get('safesplit_hist')
    if not benign_hist:
        print("  [warn] checkpoint has no SafeSplit benign history; attack-phase z-scores start cold")

    # ── Clean accuracy of the exact model the attacks will target ─────────
    client_model.eval(); server_model.eval()
    c = t = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            c += server_model(client_model(x)).argmax(1).eq(y).sum().item(); t += y.size(0)
    clean_acc = 100.0 * c / t
    print(f"\n  Clean test accuracy (SafeSplit, {Config.MODEL_NAME}, cut {Config.CUT_LAYER}): {clean_acc:.2f}%")
    pd.DataFrame([{'defense': DEFENSE_NAME, 'model': Config.MODEL_NAME, 'cut_layer': Config.CUT_LAYER,
                   'dataset': Config.DATASET, 'checkpoint': checkpoint_path, 'clean_acc': clean_acc}]
                 ).to_csv(f"{Config.RESULTS_DIR}/{DEFENSE_NAME}_clean_acc_{TAG}.csv", index=False)

    if args.skip_attacks:
        sys.exit(0)

    # ═════════════════════════════════════════════════════════════════════
    #  ATTACKS (same parameters as unified_defense_run.py / zorro_run.py)
    # ═════════════════════════════════════════════════════════════════════
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
        print("\n" + "=" * 60 + f"\n  {title} vs SAFESPLIT — {Config.MODEL_NAME} (cut {Config.CUT_LAYER})\n" + "=" * 60)
        t0 = time.time()
        try:
            results[name] = fn()
            print(f"  [done in {time.time()-t0:.0f}s]")
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
            print(f"  [FAILED] {failures[name]}")
            traceback.print_exc()

    def record_activity(name, defense, stats):
        activity[name] = stats
        pd.DataFrame(defense._detection_history).to_csv(
            f"{R}/{DEFENSE_NAME}_detections_{name.lower()}_{TAG}.csv", index=False)

    # 1 ── White-Box ────────────────────────────────────────────────────
    def attack_whitebox():
        atk = WhiteBoxInversionAttack(client_model=client_model, dataset=Config.DATASET,
                                      iterations=ITERATIONS, lr=1e-2)
        tracker, done, vis_o, vis_r = AttackMetricsTracker(), 0, None, None
        for inputs, _ in test_loader:
            if done >= MAX_IMAGES:
                break
            inputs = inputs.to(device)[:MAX_IMAGES - done]
            with torch.no_grad():
                smashed = client_model(inputs)
            rec = atk.reconstruct(smashed, inputs.shape)
            den = torch.clamp(inputs * std + mean, 0, 1)
            tracker.log_batch(den, rec)
            if vis_o is None:
                vis_o, vis_r = den[:8].cpu(), rec[:8].cpu()
            done += inputs.shape[0]
        s = tracker.get_summary()
        if vis_o is not None:
            save_ae_attack_visualization(originals=vis_o, reconstructed=vis_r, baseline_summary=s,
                                         defense_summary=s, defense_name=f"WhiteBox_SafeSplit_{TAG}",
                                         dataset=Config.DATASET, results_dir=R)
        pd.DataFrame([s]).to_csv(f"{R}/{DEFENSE_NAME}_attack_whitebox_{TAG}.csv", index=False)
        print(f"  PSNR {s['mean_psnr']:.2f} dB | SSIM {s['mean_ssim']:.4f}")
        return s

    # 2 ── UnSplit ──────────────────────────────────────────────────────
    def attack_unsplit():
        atk = UnSplitAttack(client_model=client_model, in_channels=in_channels,
                            clone_builder=build_fresh_client,
                            main_iters=Config.unsplit_main_iters,
                            input_iters=Config.unsplit_input_iters,
                            model_iters=Config.unsplit_model_iters)
        s = atk.run_attack(test_loader, num_batches=MAX_IMAGES // 32 or 1)
        src = f"{R}/{Config.MODEL_NAME.lower()}_unsplit_no_defense.png"     # hard-coded name in the attack
        dst = f"{R}/{DEFENSE_NAME}_unsplit_{TAG}.png"
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
        pd.DataFrame([s]).to_csv(f"{R}/{DEFENSE_NAME}_attack_unsplit_{TAG}.csv", index=False)
        print(f"  PSNR {s['psnr']:.2f} dB | SSIM {s['ssim']:.4f}")
        return s

    # 3 ── AE decoder ───────────────────────────────────────────────────
    def attack_ae():
        s, o, r = run_ae_decoder_attack(client_model=client_model, train_loader=train_loader,
                                        test_loader=test_loader, device=device, dataset=Config.DATASET,
                                        preprocess_fn=None, ae_epochs=50,
                                        label=f'{Config.MODEL_NAME} SafeSplit AE Decoder')
        pd.DataFrame([s]).to_csv(f"{R}/{DEFENSE_NAME}_attack_ae_decoder_{TAG}.csv", index=False)
        if len(o) > 0:
            save_ae_attack_visualization(originals=o, reconstructed=r, baseline_summary=s,
                                         defense_summary=s, defense_name=f"SafeSplit-{TAG}",
                                         dataset=Config.DATASET, results_dir=R)
        return s

    # 4 ── FSHA ─────────────────────────────────────────────────────────
    def attack_fsha():
        atk = FSHAAttack(client_model=client_model, in_channels=in_channels, dataset=Config.DATASET,
                         pilot_builder=build_fresh_client, critic_iters=CRITIC_ITERS)
        atk.hijack(train_loader, test_loader, epochs=HIJACK_EPOCHS)
        s = atk.reconstruct(train_loader, num_images=MAX_IMAGES)
        src = f"{R}/fsha_no_defense.png"                                      # hard-coded name in the attack
        dst = f"{R}/fsha_{DEFENSE_NAME}_{TAG}.png"
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
        pd.DataFrame([s]).to_csv(f"{R}/{DEFENSE_NAME}_attack_fsha_{TAG}.csv", index=False)
        return s

    # 5 ── Label leakage ────────────────────────────────────────────────
    def attack_label_leakage():
        btr, bte = build_binary_split_loaders(train_loader.dataset, target_class=POSITIVE_CLASS,
                                              positive_ratio=POSITIVE_RATIO, batch_size=LEAKAGE_BATCH)
        lc, ls = build_fresh_split(num_classes=1)
        atk = GradientNormLabelLeakageAttack(client_model=lc, server_model=ls, dataset=Config.DATASET,
                                             target_class=POSITIVE_CLASS, learning_rate=LEAKAGE_LR)
        s = atk.run(btr, bte, epochs=LEAKAGE_EPOCHS)
        atk.save_visualization(tag=f"{DEFENSE_NAME}_{TAG}")
        pd.DataFrame([s]).to_csv(f"{R}/{DEFENSE_NAME}_attack_label_leakage_{TAG}.csv", index=False)
        pd.DataFrame(atk.history).to_csv(f"{R}/{DEFENSE_NAME}_label_leakage_batches_{TAG}.csv", index=False)
        return s

    # 6 ── VILLAIN (SafeSplit monitors the injection phase, one epoch per turn) ──
    def attack_villain():
        idx_loader = build_indexed_loader(train_loader.dataset, batch_size=VILLAIN_BATCH, shuffle=True)
        vc, vs = build_fresh_split(num_classes=Config.NUM_CLASSES)
        atk = VILLAINBackdoorAttack(client_model=vc, server_model=vs, base_dataset=train_loader.dataset,
                                    dataset=Config.DATASET, num_classes=Config.NUM_CLASSES,
                                    target_label=TARGET_LABEL, beta=TRIGGER_BETA,
                                    trigger_fraction=TRIGGER_FRACTION, dropout_keep=DROPOUT_KEEP,
                                    gamma_low=GAMMA_LOW, gamma_high=GAMMA_HIGH,
                                    poison_rate=POISON_RATE, candidates_per_batch=CANDIDATES)
        atk.warmup(idx_loader, epochs=WARMUP_EPOCHS)
        base, _ = atk.evaluate(test_loader)
        print(f"  Clean accuracy before attack: {base:.2f}%")
        atk.infer_labels(idx_loader, epochs=INFERENCE_EPOCHS)
        atk.fabricate_trigger(idx_loader)

        defense = make_defense(atk.server_model, device, benign_hist)
        stats = guarded_epochs(
            atk, defense, INJECTION_EPOCHS,
            run_epoch=lambda: atk.inject_backdoor(idx_loader, test_loader, epochs=1),
            test_loader=test_loader, label='VILLAIN')
        record_activity('VILLAIN', defense, stats)

        s = atk.summarise(test_loader=test_loader, clean_baseline=base)
        atk.save_visualization(tag=f"{DEFENSE_NAME}_{TAG}")
        pd.DataFrame([{**s, **{f'safesplit_{k}': v for k, v in stats.items()}}]).to_csv(
            f"{R}/{DEFENSE_NAME}_attack_villain_{TAG}.csv", index=False)
        pd.DataFrame(atk.history).to_csv(f"{R}/{DEFENSE_NAME}_villain_epochs_{TAG}.csv", index=False)
        return s

    # 7/8 ── Backdoor poisoning (client: monitored; server: attacker IS the server) ──
    def attack_backdoor(mode):
        def _run():
            bc, bs = build_fresh_split(num_classes=Config.NUM_CLASSES)
            kw = dict(client_model=bc, server_model=bs, base_dataset=train_loader.dataset,
                      dataset=Config.DATASET, num_classes=Config.NUM_CLASSES, mode=mode,
                      target_label=BD_TARGET, poison_rate=BD_RATE, patch_size=BD_PATCH,
                      trigger_value=BD_TRIG, model_tag=f"{DEFENSE_NAME}_{Config.MODEL_NAME.lower()}_sl")
            if mode == 'server':
                kw['surrogate_builder'] = build_fresh_client
            atk = BackdoorPoisonAttack(**kw)

            extra = {}
            if mode == 'server':
                atk.pretrain_server_backdoor(test_loader, epochs=BD_SURR_EPOCHS)
                print("  [SafeSplit] NOT applicable: the attacker is the server that runs SafeSplit. "
                      "Running the attack unmonitored.")
                atk.train(train_loader, test_loader, epochs=BD_EPOCHS)
            else:
                defense = make_defense(atk.server_model, device, benign_hist)
                stats = guarded_epochs(
                    atk, defense, BD_EPOCHS,
                    run_epoch=lambda: atk.train(train_loader, test_loader, epochs=1),
                    test_loader=test_loader, label='Backdoor(Client)')
                record_activity('BackdoorPoison_Client', defense, stats)
                extra = {f'safesplit_{k}': v for k, v in stats.items()}

            s = atk.summarise()
            atk.save_visualization(tag=f"{DEFENSE_NAME}_{TAG}")
            pd.DataFrame([{**s, **extra}]).to_csv(
                f"{R}/{DEFENSE_NAME}_attack_backdoor_poison_{mode}_{TAG}.csv", index=False)
            pd.DataFrame(atk.history).to_csv(
                f"{R}/{DEFENSE_NAME}_backdoor_poison_{mode}_epochs_{TAG}.csv", index=False)
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

    # ── Consistency check: every one of the 8 attacks has a result or a recorded failure
    accounted = set(results) | set(failures)
    assert accounted == set(EXPECTED_ATTACKS), f"Attacks unaccounted for: {set(EXPECTED_ATTACKS) - accounted}"
    print(f"\n  [check] 8/8 attacks accounted for: {len(results)} succeeded, {len(failures)} failed")
    if failures:
        pd.DataFrame([{'defense': DEFENSE_NAME, 'attack': k, 'error': v} for k, v in failures.items()]
                     ).to_csv(f"{R}/{DEFENSE_NAME}_FAILURES_{TAG}.csv", index=False)

    # ═════════════════════════════════════════════════════════════════════
    #  Summary table  (— = metric not applicable; ERR = attack failed)
    # ═════════════════════════════════════════════════════════════════════
    METRICS = {
        'WhiteBox':              {'psnr': 'mean_psnr', 'ssim': 'mean_ssim'},
        'UnSplit':               {'psnr': 'psnr', 'ssim': 'ssim'},
        'AE_Decoder':            {'psnr': 'mean_psnr', 'ssim': 'mean_ssim'},
        'FSHA':                  {'psnr': 'psnr', 'ssim': 'ssim', 'mse': 'mse'},
        'LabelLeakage':          {'auc': 'q95_norm_leak_auc_cut'},
        'VILLAIN':               {'lia': 'lia', 'asr': 'asr', 'cda': 'cda'},
        'BackdoorPoison_Client': {'asr': 'asr', 'cda': 'cda'},
        'BackdoorPoison_Server': {'asr': 'asr', 'cda': 'cda'},
    }
    LABELS = {'WhiteBox': 'White-Box', 'UnSplit': 'UnSplit', 'AE_Decoder': 'AE Decoder', 'FSHA': 'FSHA',
              'LabelLeakage': 'Label Leakage', 'VILLAIN': 'VILLAIN',
              'BackdoorPoison_Client': 'Backdoor(Client)', 'BackdoorPoison_Server': 'Backdoor(Server)'}
    COLS = [('psnr', 'PSNR (dB)', 2), ('ssim', 'SSIM', 4), ('mse', 'MSE', 5),
            ('auc', 'Leak AUC', 4), ('lia', 'LIA (%)', 2), ('asr', 'ASR (%)', 2), ('cda', 'CDA (%)', 2)]

    def cell(attack, col, dec):
        if col not in METRICS[attack]:
            return BLANK
        if attack in failures:
            return "ERR"
        v = results[attack].get(METRICS[attack][col])
        return "MISSING" if v is None else f"{float(v):.{dec}f}"

    W = 96
    print("\n" + "=" * W)
    print(f"    SAFESPLIT — {Config.MODEL_NAME} | cut layer {Config.CUT_LAYER} | {Config.DATASET} | "
          f"clean accuracy {clean_acc:.2f}%")
    print("=" * W)
    print(f"{'Attack':<20}" + "".join(f"{h:>11}" for _, h, _ in COLS))
    print("-" * W)
    for a in EXPECTED_ATTACKS:
        print(f"  {LABELS[a]:<18}" + "".join(f"{cell(a, c, d):>11}" for c, _, d in COLS))
    print("=" * W)
    for a, msg in failures.items():
        print(f"  FAILED  {LABELS[a]}: {msg}")

    print("\n  SafeSplit in the loop?")
    for a in EXPECTED_ATTACKS:
        print(f"    {LABELS[a]:<18} {DEFENSE_ACTIVE[a]}")

    print("\n" + "-" * W + "\n    SAFESPLIT ACTIVITY (flags / rollbacks / skipped turns)\n" + "-" * W)
    if train_stats:
        print(f"  {'Benign training':<24} flagged {train_stats['flagged_as_poisoned']}/"
              f"{train_stats['total_clients_analyzed']} turns "
              f"(false positives), rollbacks {train_stats['rollbacks_performed']}, "
              f"skipped {train_stats['skipped_turns']}")
    for name, st in activity.items():
        print(f"  {LABELS[name]:<24} accepted {st['accepted']}, flagged/rolled back {st['rolled_back']} "
              f"(turns {st['flagged_epochs']}), skipped {st['skipped']}")
    print("-" * W)

    print("\n" + "-" * W + "\n    LABEL LEAKAGE DETAIL (95% quantile leak AUC over batches) — SAFESPLIT\n" + "-" * W)
    if 'LabelLeakage' in results:
        ll = results['LabelLeakage']
        for label, key in [('Norm (cut layer)', 'q95_norm_leak_auc_cut'),
                           ('Cosine (cut layer)', 'q95_cosine_leak_auc_cut'),
                           ('Norm (first layer)', 'q95_norm_leak_auc_first'),
                           ('Cosine (first layer)', 'q95_cosine_leak_auc_first'),
                           ('Majority counting accuracy', 'q95_majority_accuracy_cut')]:
            print(f"  {label:<28} {ll[key]:>10.4f}")
    else:
        print(f"  Label leakage failed: {failures.get('LabelLeakage')}")
    print("-" * W)

    rows = []
    for a in EXPECTED_ATTACKS:
        row = {'defense': DEFENSE_NAME, 'model': Config.MODEL_NAME, 'cut_layer': Config.CUT_LAYER,
               'dataset': Config.DATASET, 'seed': args.seed, 'clean_acc': clean_acc, 'attack': a,
               'safesplit_in_loop': DEFENSE_ACTIVE[a],
               'status': 'FAILED' if a in failures else 'ok', 'error': failures.get(a)}
        for c, _, _ in COLS:
            row[c] = None if (c not in METRICS[a] or a in failures) else results[a].get(METRICS[a][c])
        if a in activity:
            row.update({f'safesplit_{k}': str(v) for k, v in activity[a].items()})
        if a == 'LabelLeakage' and a in results:
            for k in ('q95_norm_leak_auc_cut', 'q95_cosine_leak_auc_cut', 'q95_norm_leak_auc_first',
                      'q95_cosine_leak_auc_first', 'q95_majority_accuracy_cut'):
                row[k] = results[a].get(k)
        rows.append(row)
    out = f"{R}/{DEFENSE_NAME}_all_attacks_{TAG}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Combined SafeSplit results saved → {out}")