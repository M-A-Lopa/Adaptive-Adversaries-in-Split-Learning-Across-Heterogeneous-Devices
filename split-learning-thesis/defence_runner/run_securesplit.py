
import os
import sys
import time
import random
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SecureSplitDefense prints emoji; legacy Windows code pages would raise UnicodeEncodeError.
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
from existing_defense.securesplit_defense import SecureSplitDefense

BLANK = "—"
DEFENSE_NAME = "securesplit"

# ── SecureSplit hyper-parameters = the defaults of SecureSplitDefense ───────
SS_TRANSFORM_METHOD     = 'umap_style'
SS_TARGET_DIM           = 64
SS_N_CLUSTERS           = 2
SS_VOTING_ROUNDS        = 3
SS_REJECTION_THRESHOLD  = 0.1
SS_WARMUP_BATCHES       = 30

EXPECTED_ATTACKS = ['WhiteBox', 'UnSplit', 'AE_Decoder', 'FSHA', 'LabelLeakage',
                    'VILLAIN', 'BackdoorPoison_Client', 'BackdoorPoison_Server']

# Was SecureSplit actually in the loop for this attack?
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
    """The server contains nn.LazyLinear (uninitialised until its first forward pass)."""
    was = (client.training, server.training)
    client.eval(); server.eval()
    with torch.no_grad():
        server(client(torch.zeros(2, in_channels, image_size, image_size, device=device)))
    client.train(was[0]); server.train(was[1])


# ═════════════════════════════════════════════════════════════════════════════
#  SecureSplit helpers
# ═════════════════════════════════════════════════════════════════════════════
def embedding_dim_of(client, device, in_channels=3, image_size=32):
    """True flattened embedding size of THIS client (the repo's SecureSplitManager hard-codes
    dims that do not match the models used here)."""
    was = client.training
    client.eval()
    with torch.no_grad():
        z = client(torch.zeros(1, in_channels, image_size, image_size, device=device))
    client.train(was)
    return int(z[0].numel())


def make_defense(embedding_dim, device):
    return SecureSplitDefense(embedding_dim=embedding_dim,
                              transform_method=SS_TRANSFORM_METHOD,
                              target_dim=SS_TARGET_DIM,
                              n_clusters=SS_N_CLUSTERS,
                              voting_rounds=SS_VOTING_ROUNDS,
                              rejection_threshold=SS_REJECTION_THRESHOLD,
                              warmup_batches=SS_WARMUP_BATCHES,
                              cut_layer=Config.CUT_LAYER,
                              device=str(device))


@torch.no_grad()
def prefit_defense(defense, client, loader, device):
    """Server-side warm-up on BENIGN embeddings of the deployed client, so the transformer is
    fitted (defense active) before any attacked batch arrives."""
    was = client.training
    client.eval()
    for x, y in loader:
        defense.filter(client(x.to(device)), y.to(device))
        if defense.is_active():
            break
    client.train(was)
    assert defense.is_active(), "SecureSplit warm-up did not fit the transformer"


def new_holder():
    return {'labels': None, 'keep': None, 'poison_mask': None,
            'seen': 0, 'rejected': 0,
            'poison_seen': 0, 'poison_caught': 0, 'benign_seen': 0, 'benign_rejected': 0}


def activity_stats(h):
    st = {'samples_seen': h['seen'], 'samples_rejected': h['rejected'],
          'rejection_rate_pct': 100.0 * h['rejected'] / max(1, h['seen'])}
    if h['poison_seen'] > 0:
        st.update({'poison_seen': h['poison_seen'], 'poison_caught': h['poison_caught'],
                   'poison_recall_pct': 100.0 * h['poison_caught'] / h['poison_seen'],
                   'benign_seen': h['benign_seen'], 'benign_rejected': h['benign_rejected'],
                   'benign_false_reject_pct': 100.0 * h['benign_rejected'] / max(1, h['benign_seen'])})
    return st


class FilteredServer(nn.Module):
    """Wraps an attack's server model. In training mode, filters the uploaded embeddings with
    SecureSplit (needs the batch labels, provided through `holder`) and forwards ONLY the kept
    samples. Output rows for rejected samples are zeros; the paired MaskedCriterion excludes them
    from the loss. Eval mode passes straight through."""

    def __init__(self, inner, defense, holder):
        super().__init__()
        self.inner = inner
        self.defense = defense
        self.holder = holder

    def forward(self, z):
        h = self.holder
        labels = h.get('labels')
        if (not self.training) or labels is None:
            return self.inner(z)

        _, _, rej = self.defense.filter(z.detach().contiguous(), labels)
        rej = rej.to(z.device)
        keep = ~rej
        idx = keep.nonzero(as_tuple=True)[0]

        h['seen'] += int(z.size(0))
        h['rejected'] += int(rej.sum())
        pm = h.get('poison_mask')
        if pm is not None:
            pm = pm.to(z.device)
            h['poison_seen'] += int(pm.sum())
            h['poison_caught'] += int((rej & pm).sum())
            h['benign_seen'] += int((~pm).sum())
            h['benign_rejected'] += int((rej & ~pm).sum())

        out_k = self.inner(z[idx])
        out = z.new_zeros(z.size(0), out_k.size(1)).index_copy(0, idx, out_k)
        h['keep'] = keep
        return out


class MaskedCriterion(nn.Module):
    """Loss over the kept samples only (matches true filtering: mean over kept, not over batch)."""

    def __init__(self, inner, holder):
        super().__init__()
        self.inner = inner
        self.holder = holder

    def forward(self, outputs, labels):
        keep = self.holder.get('keep')
        self.holder['keep'] = None
        if keep is None or outputs.size(0) != keep.numel():
            return self.inner(outputs, labels)
        return self.inner(outputs[keep], labels[keep])


def install_securesplit(atk, defense, holder, track_poison):
    """Put SecureSplit into an attack object's training loop WITHOUT editing the attack code:
    swap its server model / criterion for the wrappers, and intercept _split_step (to expose the
    labels) and, optionally, _poison_batch (to expose the poison mask). Returns state to undo."""
    saved = {'server': atk.server_model, 'crit': atk.criterion}
    atk.server_model = FilteredServer(atk.server_model, defense, holder)
    atk.criterion = MaskedCriterion(atk.criterion, holder)

    orig_step = atk._split_step

    def step(inputs, labels, *a, **k):
        holder['labels'] = labels
        try:
            return orig_step(inputs, labels, *a, **k)
        finally:
            holder['labels'] = None
            holder['poison_mask'] = None
            holder['keep'] = None
    atk._split_step = step

    if track_poison:
        orig_pb = atk._poison_batch

        def pb(images, labels):
            out = orig_pb(images, labels)
            holder['poison_mask'] = out[2]
            return out
        atk._poison_batch = pb
    return saved


def uninstall_securesplit(atk, saved):
    atk.server_model = saved['server']
    atk.criterion = saved['crit']
    for name in ('_split_step', '_poison_batch'):
        atk.__dict__.pop(name, None)


# ═════════════════════════════════════════════════════════════════════════════
#  SecureSplit-filtered split-learning trainer (benign client)
# ═════════════════════════════════════════════════════════════════════════════
class SecureSplitTrainer:
    def __init__(self, client, server, train_loader, test_loader, device, ckpt_path):
        self.client, self.server = client, server
        self.train_loader, self.test_loader = train_loader, test_loader
        self.device, self.ckpt_path = device, ckpt_path

        self.client_opt = torch.optim.Adam(client.parameters(), lr=Config.LEARNING_RATE)
        self.server_opt = torch.optim.Adam(server.parameters(), lr=Config.LEARNING_RATE)
        self.client_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.client_opt, patience=5, factor=0.5)
        self.server_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.server_opt, patience=5, factor=0.5)
        self.criterion = nn.CrossEntropyLoss()
        self.defense = make_defense(embedding_dim_of(client, device), device)

        self.losses, self.train_accs, self.test_accs, self.epoch_rej = [], [], [], []

    def _train_one_epoch(self, epoch_idx):
        self.client.train(); self.server.train()
        rej0, proc0 = self.defense._total_rejected, self.defense._total_processed
        run_loss, correct, total = 0.0, 0, 0
        bar = tqdm(self.train_loader, desc=f"  SecureSplit Epoch [{epoch_idx+1}/{Config.EPOCHS}]", leave=False)
        for x, y in bar:
            x, y = x.to(self.device), y.to(self.device)
            self.client_opt.zero_grad()
            z = self.client(x)
            z_leaf = z.detach().requires_grad_(True)

            # server-side SecureSplit filter (warm-up batches are accepted unchanged)
            _, _, rej = self.defense.filter(z_leaf.detach().contiguous(), y)
            keep = (~rej).to(self.device)
            idx = keep.nonzero(as_tuple=True)[0]

            self.server_opt.zero_grad()
            out = self.server(z_leaf[idx])
            loss = self.criterion(out, y[idx])
            loss.backward()
            self.server_opt.step()

            z.backward(z_leaf.grad)          # rejected rows get zero gradient
            self.client_opt.step()

            run_loss += loss.item(); total += idx.numel()
            correct += out.argmax(1).eq(y[idx]).sum().item()
            bar.set_postfix(Loss=f"{loss.item():.4f}", Acc=f"{100.*correct/max(1,total):.2f}%")

        d_rej = self.defense._total_rejected - rej0
        d_proc = self.defense._total_processed - proc0
        rej_rate = 100.0 * d_rej / max(1, d_proc)
        self.epoch_rej.append(rej_rate)
        return run_loss / len(self.train_loader), 100.0 * correct / max(1, total), rej_rate

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
        print(f"   SECURESPLIT SPLIT LEARNING — {Config.MODEL_NAME} | cut {Config.CUT_LAYER}")
        print("=" * 60)
        best = 0.0
        for epoch in range(Config.EPOCHS):
            loss, tr, rej = self._train_one_epoch(epoch)
            te = self.evaluate()
            self.losses.append(loss); self.train_accs.append(tr); self.test_accs.append(te)
            self.client_sched.step(te); self.server_sched.step(te)
            print(f"  Epoch {epoch+1:3d}/{Config.EPOCHS} | Loss {loss:.4f} | Train(kept) {tr:.2f}% | "
                  f"Test {te:.2f}% | rejected this epoch {rej:.2f}%")
            if te > best:
                best = te
                torch.save({'epoch': epoch,
                            'client_state': self.client.state_dict(),
                            'server_state': self.server.state_dict(),
                            'best_acc': best,
                            'dataset': Config.DATASET,
                            'model_type': Config.MODEL_NAME,
                            'cut_layer': Config.CUT_LAYER,
                            'defense': DEFENSE_NAME}, self.ckpt_path)
        self.defense.print_stats()
        print(f"  SecureSplit training complete. Best test accuracy: {best:.2f}%")
        return self.losses, self.train_accs, self.test_accs

    def stats(self):
        s = dict(self.defense.get_stats())
        s.update({'transform_method': SS_TRANSFORM_METHOD, 'target_dim': SS_TARGET_DIM,
                  'n_clusters': SS_N_CLUSTERS, 'voting_rounds': SS_VOTING_ROUNDS,
                  'rejection_threshold': SS_REJECTION_THRESHOLD, 'warmup_batches': SS_WARMUP_BATCHES})
        return s


def plot_training(losses, tr, te, rej, tag):
    ep = range(1, len(losses) + 1)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5))
    a1.plot(ep, losses, 'b-', lw=2); a1.set_title('Training Loss'); a1.grid(alpha=.3)
    a2.plot(ep, tr, 'b-', lw=2, label='Train (kept)'); a2.plot(ep, te, 'r-', lw=2, label='Test')
    a2.set_title('Accuracy (%)'); a2.legend(); a2.grid(alpha=.3)
    a3.plot(ep, rej, 'g-', lw=2); a3.set_title('Benign samples rejected per epoch (%)'); a3.grid(alpha=.3)
    plt.suptitle(f'SecureSplit {Config.MODEL_NAME} Split Learning — {Config.DATASET} (cut {Config.CUT_LAYER})')
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
    print(f"Device: {device} | Defense: SECURESPLIT | Model: {Config.MODEL_NAME} | "
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
        assert ck.get('defense', DEFENSE_NAME) == DEFENSE_NAME, "Checkpoint is not a SecureSplit checkpoint"
        return ck

    def build_fresh_client():
        """Untrained client (attacker-side clone for UnSplit / FSHA / server-backdoor surrogate)."""
        return build_client()

    def build_fresh_split(num_classes):
        """Client initialised from the SecureSplit-trained checkpoint; server only if 10-class."""
        c, s = build_client(), build_server(num_classes)
        ck = _load_ckpt()
        load_checked(c, ck['client_state'], 'securesplit client')
        if num_classes == Config.NUM_CLASSES:
            load_checked(s, ck['server_state'], 'securesplit server')
        return c, s

    client_model, server_model = build_client(), build_server(Config.NUM_CLASSES)
    materialize_lazy(client_model, server_model, device, in_channels)
    EMB_DIM = embedding_dim_of(client_model, device, in_channels)
    print(f"  Embedding dim at cut {Config.CUT_LAYER}: {EMB_DIM}")

    train_stats = None
    if os.path.exists(checkpoint_path):
        print(f"\n[✓] Found SecureSplit checkpoint: {checkpoint_path} — skipping training")
    else:
        print(f"\n[!] No checkpoint at {checkpoint_path} — training with SecureSplit")
        trainer = SecureSplitTrainer(client_model, server_model, train_loader, test_loader, device, checkpoint_path)
        losses, tr_accs, te_accs = trainer.train()
        plot_training(losses, tr_accs, te_accs, trainer.epoch_rej, TAG)
        pd.DataFrame({'epoch': range(1, len(losses) + 1), 'train_loss': losses,
                      'train_accuracy_kept': tr_accs, 'test_accuracy': te_accs,
                      'benign_rejected_pct': trainer.epoch_rej}
                     ).to_csv(f"{Config.RESULTS_DIR}/{DEFENSE_NAME}_results_{TAG}.csv", index=False)
        train_stats = trainer.stats()
        pd.DataFrame([train_stats]).to_csv(f"{Config.RESULTS_DIR}/{DEFENSE_NAME}_stats_{TAG}.csv", index=False)

    # always evaluate/attack the BEST checkpoint (the one saved by the trainer)
    ck = _load_ckpt()
    load_checked(client_model, ck['client_state'], 'securesplit client')
    load_checked(server_model, ck['server_state'], 'securesplit server')

    # ── Clean accuracy of the exact model the attacks will target ─────────
    client_model.eval(); server_model.eval()
    c = t = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            c += server_model(client_model(x)).argmax(1).eq(y).sum().item(); t += y.size(0)
    clean_acc = 100.0 * c / t
    print(f"\n  Clean test accuracy (SecureSplit, {Config.MODEL_NAME}, cut {Config.CUT_LAYER}): {clean_acc:.2f}%")
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
        print("\n" + "=" * 60 + f"\n  {title} vs SECURESPLIT — {Config.MODEL_NAME} (cut {Config.CUT_LAYER})\n" + "=" * 60)
        t0 = time.time()
        try:
            results[name] = fn()
            print(f"  [done in {time.time()-t0:.0f}s]")
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
            print(f"  [FAILED] {failures[name]}")
            traceback.print_exc()

    def record_activity(name, holder):
        st = activity_stats(holder)
        activity[name] = st
        pd.DataFrame([st]).to_csv(f"{R}/{DEFENSE_NAME}_activity_{name.lower()}_{TAG}.csv", index=False)
        print(f"  [SecureSplit] {name}: rejected {st['samples_rejected']}/{st['samples_seen']} "
              f"({st['rejection_rate_pct']:.2f}%)"
              + (f" | poisoned caught {st['poison_caught']}/{st['poison_seen']} "
                 f"({st['poison_recall_pct']:.1f}%), benign wrongly rejected "
                 f"{st['benign_false_reject_pct']:.2f}%" if 'poison_seen' in st else ""))

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
                                         defense_summary=s, defense_name=f"WhiteBox_SecureSplit_{TAG}",
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
                                        label=f'{Config.MODEL_NAME} SecureSplit AE Decoder')
        pd.DataFrame([s]).to_csv(f"{R}/{DEFENSE_NAME}_attack_ae_decoder_{TAG}.csv", index=False)
        if len(o) > 0:
            save_ae_attack_visualization(originals=o, reconstructed=r, baseline_summary=s,
                                         defense_summary=s, defense_name=f"SecureSplit-{TAG}",
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

    # 6 ── VILLAIN (SecureSplit filters the injection phase) ────────────
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

        # server fits SecureSplit on benign embeddings of the client as it is now, then filters injection
        defense = make_defense(EMB_DIM, device)
        prefit_defense(defense, atk.client_model, train_loader, device)
        holder = new_holder()
        saved = install_securesplit(atk, defense, holder, track_poison=False)
        try:
            atk.inject_backdoor(idx_loader, test_loader, epochs=INJECTION_EPOCHS)
        finally:
            uninstall_securesplit(atk, saved)
        record_activity('VILLAIN', holder)

        s = atk.summarise(test_loader=test_loader, clean_baseline=base)
        atk.save_visualization(tag=f"{DEFENSE_NAME}_{TAG}")
        pd.DataFrame([{**s, **{f'securesplit_{k}': v for k, v in activity['VILLAIN'].items()}}]).to_csv(
            f"{R}/{DEFENSE_NAME}_attack_villain_{TAG}.csv", index=False)
        pd.DataFrame(atk.history).to_csv(f"{R}/{DEFENSE_NAME}_villain_epochs_{TAG}.csv", index=False)
        return s

    # 7/8 ── Backdoor poisoning (client: filtered; server: attacker IS the server) ──
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
                print("  [SecureSplit] NOT applicable: the attacker is the server that runs SecureSplit. "
                      "Running the attack unmonitored.")
                atk.train(train_loader, test_loader, epochs=BD_EPOCHS)
            else:
                defense = make_defense(EMB_DIM, device)
                prefit_defense(defense, atk.client_model, train_loader, device)
                holder = new_holder()
                saved = install_securesplit(atk, defense, holder, track_poison=True)
                try:
                    atk.train(train_loader, test_loader, epochs=BD_EPOCHS)
                finally:
                    uninstall_securesplit(atk, saved)
                record_activity('BackdoorPoison_Client', holder)
                extra = {f'securesplit_{k}': v for k, v in activity['BackdoorPoison_Client'].items()}

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
    print(f"    SECURESPLIT — {Config.MODEL_NAME} | cut layer {Config.CUT_LAYER} | {Config.DATASET} | "
          f"clean accuracy {clean_acc:.2f}%")
    print("=" * W)
    print(f"{'Attack':<20}" + "".join(f"{h:>11}" for _, h, _ in COLS))
    print("-" * W)
    for a in EXPECTED_ATTACKS:
        print(f"  {LABELS[a]:<18}" + "".join(f"{cell(a, c, d):>11}" for c, _, d in COLS))
    print("=" * W)
    for a, msg in failures.items():
        print(f"  FAILED  {LABELS[a]}: {msg}")

    print("\n  SecureSplit in the loop?")
    for a in EXPECTED_ATTACKS:
        print(f"    {LABELS[a]:<18} {DEFENSE_ACTIVE[a]}")

    print("\n" + "-" * W + "\n    SECURESPLIT ACTIVITY (embeddings rejected by the filter)\n" + "-" * W)
    if train_stats:
        print(f"  {'Benign training':<24} rejected {train_stats['total_rejected']}/"
              f"{train_stats['total_processed']} samples "
              f"({train_stats['rejection_rate_pct']:.2f}%) — all were benign, i.e. false rejections")
    for name, st in activity.items():
        line = (f"  {LABELS[name]:<24} rejected {st['samples_rejected']}/{st['samples_seen']} "
                f"({st['rejection_rate_pct']:.2f}%)")
        if 'poison_seen' in st:
            line += (f" | poisoned caught {st['poison_caught']}/{st['poison_seen']} "
                     f"({st['poison_recall_pct']:.1f}%), benign wrongly rejected "
                     f"{st['benign_false_reject_pct']:.2f}%")
        print(line)
    print("-" * W)

    print("\n" + "-" * W + "\n    LABEL LEAKAGE DETAIL (95% quantile leak AUC over batches) — SECURESPLIT\n" + "-" * W)
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
               'securesplit_in_loop': DEFENSE_ACTIVE[a],
               'status': 'FAILED' if a in failures else 'ok', 'error': failures.get(a)}
        for c, _, _ in COLS:
            row[c] = None if (c not in METRICS[a] or a in failures) else results[a].get(METRICS[a][c])
        if a in activity:
            row.update({f'securesplit_{k}': v for k, v in activity[a].items()})
        if a == 'LabelLeakage' and a in results:
            for k in ('q95_norm_leak_auc_cut', 'q95_cosine_leak_auc_cut', 'q95_norm_leak_auc_first',
                      'q95_cosine_leak_auc_first', 'q95_majority_accuracy_cut'):
                row[k] = results[a].get(k)
        rows.append(row)
    out = f"{R}/{DEFENSE_NAME}_all_attacks_{TAG}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Combined SecureSplit results saved → {out}")