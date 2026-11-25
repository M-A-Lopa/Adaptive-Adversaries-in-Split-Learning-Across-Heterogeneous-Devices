
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from config import Config
from util.split_env import (build_split_models, get_targets, raw_images_float, make_loader,
                            set_seed, train_split_model, evaluate_accuracy, env_checkpoint_path,
                            try_load_checkpoint, smashed_shape)

PROPERTIES = {'CIFAR10': ['brightness', 'contrast'], 'MNIST': ['ink', 'brightness', 'contrast']}


def property_statistic(images01, prop):
    """images01: float array (N, H, W) or (N, H, W, C) in [0,1]."""
    flat = images01.reshape(len(images01), -1)
    if prop == 'brightness':
        return flat.mean(1)
    if prop == 'contrast':
        return flat.std(1)
    if prop == 'ink':
        return (flat > 0.5).mean(1)
    raise ValueError(f"unknown property {prop}")


def label_independent_property(ds, prop, reference_ds=None):
   
    stat = property_statistic(raw_images_float(ds), prop)
    y = get_targets(ds)
    ref = reference_ds if reference_ds is not None else ds
    ref_stat = property_statistic(raw_images_float(ref), prop)
    ref_y = get_targets(ref)
    out = np.zeros(len(y), dtype=np.int64)
    for c in np.unique(y):
        med = np.median(ref_stat[ref_y == c])
        out[y == c] = (stat[y == c] > med).astype(np.int64)
    return out


class InferenceModel(nn.Module):
    """F in Eq. (1): maps smashed data T(x) to the binary property."""

    def __init__(self, in_dim, arch='mlp', hidden=256):
        super().__init__()
        if arch == 'linear':
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, 2))
        else:
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, hidden), nn.ReLU(),
                                     nn.Dropout(0.5), nn.Linear(hidden, 2))

    def forward(self, z):
        return self.net(z)


class MaoPropertyInference:

    def __init__(self, model_name, cut_layer, dataset, raw, device, prop,
                 n_aux=5000, n_eval=10000, arch='mlp', attack_epochs=30, attack_lr=1e-3,
                 attack_batch=64, victim_epochs=None, seed=0, client_forward=None,
                 tag='no_defense', cache_dir=None):
        self.model_name, self.cut, self.dataset, self.prop = model_name, cut_layer, dataset, prop
        self.raw, self.device = raw, device
        self.n_aux, self.n_eval, self.arch = n_aux, n_eval, arch
        self.attack_epochs, self.attack_lr, self.attack_batch = attack_epochs, attack_lr, attack_batch
        self.victim_epochs = victim_epochs or Config.EPOCHS
        self.seed, self.tag = seed, tag
        self.client_forward = client_forward
        self.cache_dir = cache_dir or os.path.join(Config.SAVE_DIR, 'property')
        os.makedirs(self.cache_dir, exist_ok=True)

        print("\n" + "=" * 64)
        print("   PROPERTY INFERENCE -- Mao et al. (Eq. 1)")
        print("=" * 64)
        print(f"  Model / cut / data : {model_name} / {cut_layer} / {dataset}")
        print(f"  Property           : {prop} (binarised at the per-class median -> chance 50%)")
        print(f"  Aux (attacker)     : {n_aux} test records | eval on {n_eval} private train records")
        print(f"  Inference model F  : {arch}")


    def get_victim(self):
        """Clean split model (thesis checkpoint if it matches, else train & cache)."""
        set_seed(self.seed)
        client, server = build_split_models(self.model_name, self.cut, self.dataset, device=self.device)
        own = os.path.join(self.cache_dir, f"victim_{self.model_name.lower()}_cut{self.cut}_{self.dataset}"
                                           f"_ep{self.victim_epochs}_seed{self.seed}.pth")
        if try_load_checkpoint(client, server, env_checkpoint_path(self.model_name, self.dataset),
                               self.cut, self.device):
            pass
        elif os.path.exists(own):
            ck = torch.load(own, map_location=self.device)
            client.load_state_dict(ck['client_state']); server.load_state_dict(ck['server_state'])
            print(f"  [OK] loaded cached victim {own}")
        else:
            print("  Training the victim split model (thesis procedure)...")
            train_split_model(client, server,
                              make_loader(self.raw['train_aug'], shuffle=True, drop_last=True),
                              make_loader(self.raw['test_plain']), self.victim_epochs, self.device,
                              desc='victim')
            torch.save({'client_state': client.state_dict(), 'server_state': server.state_dict(),
                        'cut_layer': self.cut}, own)
        self.client, self.server = client, server
        self.victim_acc = evaluate_accuracy(client, server, make_loader(self.raw['test_plain']), self.device)
        print(f"  Victim test accuracy: {self.victim_acc:.2f}%")


    @torch.no_grad()
    def _smashed(self, ds, indices):
        """T(x): what the server receives for these records."""
        self.client.eval()
        fwd = self.client_forward or self.client
        out = []
        for x, _ in make_loader(ds, indices, batch_size=256):
            out.append(fwd(x.to(self.device)).float().cpu())
        return torch.cat(out)

    def run(self):
        self.get_victim()
        rng = np.random.RandomState(self.seed)
        test_ds, train_ds = self.raw['test_plain'], self.raw['train_plain']
        prop_test = label_independent_property(test_ds, self.prop, reference_ds=train_ds)
        prop_train = label_independent_property(train_ds, self.prop)

        aux_idx = rng.choice(len(test_ds), size=min(self.n_aux, len(test_ds)), replace=False)
        eval_idx = rng.choice(len(train_ds), size=min(self.n_eval, len(train_ds)), replace=False)
        Z_aux, l_aux = self._smashed(test_ds, aux_idx), torch.from_numpy(prop_test[aux_idx])
        Z_eval, l_eval = self._smashed(train_ds, eval_idx), torch.from_numpy(prop_train[eval_idx])

        # ---- Eq. (1): train F on (T(x_i), l_i) of the auxiliary records
        set_seed(self.seed)
        in_dim = int(np.prod(smashed_shape(self.client, self.dataset, self.device)))
        Fm = InferenceModel(in_dim, self.arch).to(self.device)
        opt = optim.Adam(Fm.parameters(), lr=self.attack_lr)
        n = len(l_aux)
        for ep in range(self.attack_epochs):
            Fm.train()
            perm = torch.randperm(n)
            tot = 0.0
            for b in range(0, n, self.attack_batch):
                idx = perm[b:b + self.attack_batch]
                opt.zero_grad()
                loss = F.cross_entropy(Fm(Z_aux[idx].to(self.device)), l_aux[idx].to(self.device))
                loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  F epoch {ep + 1:3d}/{self.attack_epochs} | loss {tot / n:.4f}")

      
        Fm.eval()
        with torch.no_grad():
            logits = torch.cat([Fm(Z_eval[b:b + 512].to(self.device)).cpu() for b in range(0, len(l_eval), 512)])
            aux_logits = torch.cat([Fm(Z_aux[b:b + 512].to(self.device)).cpu() for b in range(0, n, 512)])
        prob = F.softmax(logits, 1)[:, 1].numpy()
        acc = 100.0 * float((logits.argmax(1) == l_eval).float().mean())
        majority = 100.0 * max(float(l_eval.float().mean()), 1 - float(l_eval.float().mean()))
        res = {
            'attack': 'property_inference_mao', 'defense': self.tag, 'property': self.prop,
            'model': self.model_name, 'cut_layer': self.cut, 'dataset': self.dataset,
            'smashed_dim': in_dim, 'inference_model': self.arch, 'n_aux': n, 'n_eval': len(l_eval),
            'victim_test_acc': self.victim_acc,
            'pia_accuracy': acc,
            'pia_auc': float(roc_auc_score(l_eval.numpy(), prob)),
            'pia_aux_train_acc': 100.0 * float((aux_logits.argmax(1) == l_aux).float().mean()),
            'majority_baseline': majority,
            'advantage_over_chance': acc - majority,
        }
        print("\n" + "=" * 64)
        print("   PROPERTY INFERENCE RESULTS")
        print("=" * 64)
        print(f"  Attack accuracy on private data : {acc:.2f}%  (chance {majority:.2f}%)")
        print(f"  Attack AUC                      : {res['pia_auc']:.4f}")
        print("=" * 64)
        with open(os.path.join(self.cache_dir,
                               f"result_{self.model_name.lower()}_cut{self.cut}_{self.dataset}_{self.prop}_{self.tag}.json"), 'w') as f:
            json.dump(res, f, indent=2)
        return res