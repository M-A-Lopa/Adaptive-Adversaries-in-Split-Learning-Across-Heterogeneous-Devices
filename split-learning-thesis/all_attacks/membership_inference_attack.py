
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve

from config import Config
from util.split_env import (build_split_models, get_targets, make_loader, set_seed,
                            train_split_model, evaluate_accuracy)

PAPER_NUM_SHADOWS = {'CIFAR10': 100, 'MNIST': 50}
PAPER_TARGET_SIZE = {'CIFAR10': 10000, 'MNIST': 10000}


class AttackMLP(nn.Module):
    """Shokri attack model: one hidden layer of 64 ReLU units, softmax over {out, in}."""

    def __init__(self, num_classes, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_classes, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, p):
        return self.net(p)


def default_predict_fn(client, server, x):
    return F.softmax(server(client(x)), dim=1)


def default_train_fn(client, server, train_loader, epochs, device, desc):
    return train_split_model(client, server, train_loader, None, epochs, device,
                             scheduler_on='train_loss', desc=desc)


class ShokriMembershipInference:

    def __init__(self, model_name, cut_layer, dataset, raw, device,
                 target_size=None, num_shadows=None, epochs=None,
                 attack_epochs=80, attack_lr=1e-3, attack_batch=128, attack_hidden=64,
                 seed=0, cache_dir=None, train_fn=None, predict_fn=None, tag='no_defense'):
        self.model_name, self.cut, self.dataset = model_name, cut_layer, dataset
        self.raw, self.device = raw, device
        self.num_classes = Config.NUM_CLASSES
        self.N = target_size or PAPER_TARGET_SIZE[dataset]
        self.k = num_shadows if num_shadows is not None else PAPER_NUM_SHADOWS[dataset]
        self.epochs = epochs or Config.EPOCHS
        self.attack_epochs, self.attack_lr = attack_epochs, attack_lr
        self.attack_batch, self.attack_hidden = attack_batch, attack_hidden
        self.seed, self.tag = seed, tag
        self.train_fn = train_fn or default_train_fn
        self.predict_fn = predict_fn or default_predict_fn
        self.cache_dir = cache_dir or os.path.join(
            Config.SAVE_DIR, 'mia',
            f"{model_name.lower()}_cut{cut_layer}_{dataset}_{tag}_N{self.N}_ep{self.epochs}_seed{seed}")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.splits = self._make_splits()

        print("\n" + "=" * 64)
        print("   MEMBERSHIP INFERENCE -- Shokri et al., S&P 2017")
        print("=" * 64)
        print(f"  Model / cut / data : {model_name} / {cut_layer} / {dataset}")
        print(f"  Target train size  : {self.N} (members) + {self.N} non-members")
        print(f"  Shadow models      : {self.k} (paper: {PAPER_NUM_SHADOWS[dataset]})")
        print(f"  Shadow data pool   : {len(self.splits['shadow_pool'])} records (disjoint from target)")
        print(f"  Cache              : {self.cache_dir}")

    # ------------------------------------------------------------------ data
    def _make_splits(self):
        """Disjoint target-in / target-out / shadow pool from the training set."""
        targets = get_targets(self.raw['train_plain'])
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(len(targets))
        need = 2 * self.N
        if len(perm) < need + 2 * self.N:
            raise ValueError(f"Not enough data for target size {self.N}: need >= {4 * self.N} records")
        return {'target_in': perm[:self.N],
                'target_out': perm[self.N:need],
                'shadow_pool': perm[need:]}

    def _shadow_indices(self, i):
        rng = np.random.RandomState(self.seed + 1000 + i)
        pool = self.splits['shadow_pool']
        chosen = rng.choice(pool, size=2 * self.N, replace=False)
        return chosen[:self.N], chosen[self.N:]

    # --------------------------------------------------------------- models
    def _train_model(self, train_idx, desc, seed):
        set_seed(seed)
        client, server = build_split_models(self.model_name, self.cut, self.dataset, device=self.device)
        # drop_last avoids a size-1 BatchNorm batch; reshuffling every epoch still covers all members
        loader = make_loader(self.raw['train_aug'], train_idx, shuffle=True, drop_last=True)
        self.train_fn(client, server, loader, self.epochs, self.device, desc)
        return client, server

    @torch.no_grad()
    def _predictions(self, client, server, indices):
        client.eval(); server.eval()
        loader = make_loader(self.raw['train_plain'], indices, batch_size=256)
        probs, labels = [], []
        for x, y in loader:
            probs.append(self.predict_fn(client, server, x.to(self.device)).float().cpu())
            labels.append(y)
        return torch.cat(probs).numpy(), torch.cat(labels).numpy()

    def train_target(self):
        path = os.path.join(self.cache_dir, 'target.pth')
        client, server = build_split_models(self.model_name, self.cut, self.dataset, device=self.device)
        if os.path.exists(path):
            ck = torch.load(path, map_location=self.device)
            client.load_state_dict(ck['client']); server.load_state_dict(ck['server'])
            print(f"  [OK] loaded cached target {path}")
        else:
            client, server = self._train_model(self.splits['target_in'], 'target', self.seed)
            torch.save({'client': client.state_dict(), 'server': server.state_dict()}, path)
        self.target = (client, server)
        return client, server

    def build_attack_dataset(self):
        """Train k shadow models and record (label, prediction, in/out). Cached per shadow."""
        feats, labels, member = [], [], []
        for i in range(self.k):
            path = os.path.join(self.cache_dir, f'shadow_{i:03d}.npz')
            if os.path.exists(path):
                d = np.load(path)
            else:
                print(f"\n  -- shadow model {i + 1}/{self.k} --")
                in_idx, out_idx = self._shadow_indices(i)
                c, s = self._train_model(in_idx, f'shadow {i + 1}/{self.k}', self.seed + 1 + i)
                p_in, y_in = self._predictions(c, s, in_idx)
                p_out, y_out = self._predictions(c, s, out_idx)
                d = {'p': np.concatenate([p_in, p_out]), 'y': np.concatenate([y_in, y_out]),
                     'm': np.concatenate([np.ones(len(y_in)), np.zeros(len(y_out))])}
                np.savez(path, **d)
                del c, s
            feats.append(d['p']); labels.append(d['y']); member.append(d['m'])
        self.attack_X = np.concatenate(feats).astype(np.float32)
        self.attack_y = np.concatenate(labels).astype(np.int64)
        self.attack_m = np.concatenate(member).astype(np.int64)
        print(f"  Attack training set: {len(self.attack_m)} records from {self.k} shadow models")

    # --------------------------------------------------------- attack models
    def train_attack_models(self):
        torch.manual_seed(self.seed)
        self.attack_models = {}
        for c in range(self.num_classes):
            sel = self.attack_y == c
            X = torch.from_numpy(self.attack_X[sel]).to(self.device)
            m = torch.from_numpy(self.attack_m[sel]).to(self.device)
            model = AttackMLP(self.num_classes, self.attack_hidden).to(self.device)
            opt = optim.Adam(model.parameters(), lr=self.attack_lr)
            n = len(m)
            for _ in range(self.attack_epochs):
                perm = torch.randperm(n, device=self.device)
                for b in range(0, n, self.attack_batch):
                    idx = perm[b:b + self.attack_batch]
                    opt.zero_grad()
                    F.cross_entropy(model(X[idx]), m[idx]).backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                acc = (model(X).argmax(1) == m).float().mean().item() if n else float('nan')
            print(f"  attack model class {c}: {n} shadow records, train acc {100 * acc:.2f}%")
            self.attack_models[c] = model

    @torch.no_grad()
    def _score(self, probs, labels):
        """P(member) from the attack model of each record's true class."""
        scores = np.zeros(len(labels), dtype=np.float64)
        for c in range(self.num_classes):
            sel = labels == c
            if sel.any():
                x = torch.from_numpy(probs[sel].astype(np.float32)).to(self.device)
                scores[sel] = F.softmax(self.attack_models[c](x), dim=1)[:, 1].cpu().numpy()
        return scores

    # ------------------------------------------------------------- evaluate
    def evaluate(self):
        client, server = self.target
        p_in, y_in = self._predictions(client, server, self.splits['target_in'])
        p_out, y_out = self._predictions(client, server, self.splits['target_out'])
        s = np.concatenate([self._score(p_in, y_in), self._score(p_out, y_out)])
        truth = np.concatenate([np.ones(len(y_in)), np.zeros(len(y_out))])
        labels = np.concatenate([y_in, y_out])
        pred = (s >= 0.5).astype(int)

        tp = int(((pred == 1) & (truth == 1)).sum()); fp = int(((pred == 1) & (truth == 0)).sum())
        fn = int(((pred == 0) & (truth == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else float('nan')
        recall = tp / (tp + fn) if tp + fn else float('nan')
        per_class_prec = []
        for c in range(self.num_classes):
            sel = labels == c
            tpc = int(((pred == 1) & (truth == 1) & sel).sum()); fpc = int(((pred == 1) & (truth == 0) & sel).sum())
            if tpc + fpc:
                per_class_prec.append(tpc / (tpc + fpc))
        fpr, tpr, _ = roc_curve(truth, s)

        def tpr_at(f):
            return float(np.interp(f, fpr, tpr))

        eval_in = make_loader(self.raw['train_plain'], self.splits['target_in'], batch_size=256)
        eval_out = make_loader(self.raw['train_plain'], self.splits['target_out'], batch_size=256)
        res = {
            'attack': 'membership_inference_shokri', 'defense': self.tag,
            'model': self.model_name, 'cut_layer': self.cut, 'dataset': self.dataset,
            'target_size': self.N, 'num_shadows': self.k, 'epochs': self.epochs,
            'target_train_acc': evaluate_accuracy(client, server, eval_in, self.device),
            'target_nonmember_acc': evaluate_accuracy(client, server, eval_out, self.device),
            'mia_accuracy': 100.0 * float((pred == truth).mean()),
            'mia_precision': 100.0 * precision, 'mia_recall': 100.0 * recall,
            'mia_median_class_precision': 100.0 * float(np.median(per_class_prec)) if per_class_prec else float('nan'),
            'mia_auc': float(roc_auc_score(truth, s)),
            'mia_tpr_at_0.1fpr': 100.0 * tpr_at(0.001), 'mia_tpr_at_1fpr': 100.0 * tpr_at(0.01),
        }
        res['generalization_gap'] = res['target_train_acc'] - res['target_nonmember_acc']
        with open(os.path.join(self.cache_dir, 'result.json'), 'w') as f:
            json.dump(res, f, indent=2)

        print("\n" + "=" * 64)
        print("   MEMBERSHIP INFERENCE RESULTS (random guess = 50%)")
        print("=" * 64)
        print(f"  Target acc members / non-members : {res['target_train_acc']:.2f}% / {res['target_nonmember_acc']:.2f}%")
        print(f"  Attack accuracy                  : {res['mia_accuracy']:.2f}%")
        print(f"  Precision / recall (members)     : {res['mia_precision']:.2f}% / {res['mia_recall']:.2f}%")
        print(f"  Median per-class precision       : {res['mia_median_class_precision']:.2f}%")
        print(f"  AUC                              : {res['mia_auc']:.4f}")
        print(f"  TPR @ 0.1% / 1% FPR              : {res['mia_tpr_at_0.1fpr']:.2f}% / {res['mia_tpr_at_1fpr']:.2f}%")
        print("=" * 64)
        return res

    def run(self):
        self.train_target()
        self.build_attack_dataset()
        self.train_attack_models()
        return self.evaluate()