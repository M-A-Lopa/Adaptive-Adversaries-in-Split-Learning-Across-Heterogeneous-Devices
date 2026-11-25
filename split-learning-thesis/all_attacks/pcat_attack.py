import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config
from util.split_env import (build_split_models, build_client, get_targets, make_loader, stack_images,
                            set_seed, evaluate_accuracy)

PAPER_AUX_PER_CLASS = {'CIFAR10': 250, 'MNIST': 5}


class PCATAttack:

    def __init__(self, model_name, cut_layer, dataset, raw, device, aux_per_class=None,
                 public_fraction=0.1, start_batch=100, epochs=None, lr=None, vanilla_iters=5000,
                 batch_size=None, seed=0, defense_hook=None, tag='no_defense', cache_dir=None):
        self.model_name, self.cut, self.dataset = model_name, cut_layer, dataset
        self.raw, self.device = raw, device
        self.aux_per_class = aux_per_class or PAPER_AUX_PER_CLASS[dataset]
        self.public_fraction, self.start_batch = public_fraction, start_batch
        self.epochs = epochs or Config.EPOCHS
        self.lr = lr or Config.LEARNING_RATE
        self.vanilla_iters = vanilla_iters
        self.batch_size = batch_size or Config.BATCH_SIZE
        self.seed, self.tag = seed, tag
        self.defense_hook = defense_hook or (lambda s: s)
        self.num_classes = Config.NUM_CLASSES
        self.cache_dir = cache_dir or os.path.join(Config.SAVE_DIR, 'pcat')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.criterion = nn.CrossEntropyLoss()
        self._make_splits()

        print("\n" + "=" * 64)
        print("   PCAT -- pseudo-client functionality stealing (USENIX Sec '23)")
        print("=" * 64)
        print(f"  Model / cut / data : {model_name} / {cut_layer} / {dataset}")
        print(f"  Private / public   : {len(self.private_idx)} / {len(self.public_idx)} (1:9 split)")
        print(f"  Server aux data    : {len(self.aux_idx)} records ({self.aux_per_class}/class, "
              f"{100 * len(self.aux_idx) / len(self.private_idx):.2f}% of private)")
        print(f"  Late start batch   : {self.start_batch}")

    def _make_splits(self):
        targets = get_targets(self.raw['train_plain'])
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(len(targets))
        n_pub = int(round(self.public_fraction * len(perm)))
        self.public_idx, self.private_idx = perm[:n_pub], perm[n_pub:]
        aux = []
        for c in range(self.num_classes):
            pool = self.public_idx[targets[self.public_idx] == c]
            aux.extend(rng.choice(pool, size=min(self.aux_per_class, len(pool)), replace=False).tolist())
        self.aux_idx = np.asarray(aux)
        self.X_aux, self.y_aux = stack_images(self.raw['train_plain'], self.aux_idx)
        self.X_aux, self.y_aux = self.X_aux.to(self.device), self.y_aux.to(self.device)
        self.aux_by_class = {c: torch.nonzero(self.y_aux == c).flatten() for c in range(self.num_classes)}

    def _aligned_aux_batch(self, y):
        idx = torch.empty_like(y)
        for c in torch.unique(y).tolist():
            pos = torch.nonzero(y == c).flatten()
            pool = self.aux_by_class.get(c)
            if pool is None or pool.numel() == 0:
                pool = torch.arange(len(self.y_aux), device=self.device)
            idx[pos] = pool[torch.randint(len(pool), (len(pos),), device=self.device)]
        return self.X_aux[idx], self.y_aux[idx]

    @torch.no_grad()
    def _evaluate(self, F_c, F_s, f_tilde):
        loader = make_loader(self.raw['test_plain'], batch_size=256)
        F_c.eval(); F_s.eval(); f_tilde.eval()
        cv = cs = n = 0
        mse = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            s_real = self.defense_hook(F_c(x))
            s_fake = f_tilde(x)
            cv += F_s(s_real).argmax(1).eq(y).sum().item()
            cs += F_s(s_fake).argmax(1).eq(y).sum().item()
            mse += torch.mean((s_fake - s_real) ** 2, dim=tuple(range(1, s_real.dim()))).sum().item()
            n += y.numel()
        return 100.0 * cv / n, 100.0 * cs / n, mse / n

    def run_pcat(self):
        set_seed(self.seed)
        F_c, F_s = build_split_models(self.model_name, self.cut, self.dataset, device=self.device)
        set_seed(self.seed + 1)
        f_tilde = build_client(self.model_name, self.cut, self.dataset, device=self.device)

        c_opt = optim.Adam(F_c.parameters(), lr=self.lr)
        s_opt = optim.Adam(F_s.parameters(), lr=self.lr)
        p_opt = optim.Adam(f_tilde.parameters(), lr=self.lr)
        c_sch = optim.lr_scheduler.ReduceLROnPlateau(c_opt, mode='max', patience=5, factor=0.5)
        s_sch = optim.lr_scheduler.ReduceLROnPlateau(s_opt, mode='max', patience=5, factor=0.5)
        loader = make_loader(self.raw['train_aug'], self.private_idx, batch_size=self.batch_size,
                             shuffle=True, drop_last=True)
        p_params = list(f_tilde.parameters())
        step, history = 0, []

        for ep in range(self.epochs):
            F_c.train(); F_s.train(); f_tilde.train()
            run_l, run_lt, n_lt = 0.0, 0.0, 0
            for x, y in tqdm(loader, desc=f"  PCAT epoch {ep + 1}/{self.epochs}", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                c_opt.zero_grad()
                smashed = self.defense_hook(F_c(x))
                smashed_srv = smashed.detach().requires_grad_(True)
                s_opt.zero_grad()
                L = self.criterion(F_s(smashed_srv), y)
                L.backward()
                if step >= self.start_batch:
                    X_i, y_i = self._aligned_aux_batch(y)
                    F_s.eval()
                    L_t = self.criterion(F_s(f_tilde(X_i)), y_i)
                    grads = torch.autograd.grad(L_t, p_params)
                    F_s.train()
                    p_opt.zero_grad()
                    for p, g in zip(p_params, grads):
                        p.grad = g
                    p_opt.step()
                    run_lt += L_t.item(); n_lt += 1
                s_opt.step()
                smashed.backward(smashed_srv.grad)
                c_opt.step()
                run_l += L.item(); step += 1

            v_acc, s_acc, mse = self._evaluate(F_c, F_s, f_tilde)
            c_sch.step(v_acc); s_sch.step(v_acc)
            history.append({'epoch': ep + 1, 'victim_loss': run_l / len(loader),
                            'pseudo_loss': run_lt / n_lt if n_lt else float('nan'),
                            'victim_acc': v_acc, 'stolen_acc': s_acc, 'fidelity_mse': mse})
            print(f"  epoch {ep + 1:3d}/{self.epochs} | victim {v_acc:.2f}% | stolen {s_acc:.2f}% | "
                  f"gap {v_acc - s_acc:.2f} | MSE {mse:.4f}")

        self.F_c, self.F_s, self.f_tilde, self.history = F_c, F_s, f_tilde, history
        torch.save({'victim_client': F_c.state_dict(), 'victim_server': F_s.state_dict(),
                    'pseudo_client': f_tilde.state_dict(), 'history': history, 'cut_layer': self.cut},
                   os.path.join(self.cache_dir, f"pcat_{self.model_name.lower()}_cut{self.cut}_{self.dataset}_{self.tag}.pth"))
        return history[-1]

    def run_vanilla_pcat(self):
        set_seed(self.seed + 2)
        f_tilde = build_client(self.model_name, self.cut, self.dataset, device=self.device)
        F_s = self.F_s
        F_s.eval()
        for p in F_s.parameters():
            p.requires_grad_(False)
        opt = optim.Adam(f_tilde.parameters(), lr=self.lr)
        f_tilde.train()
        n = len(self.y_aux)
        for it in tqdm(range(self.vanilla_iters), desc="  Vanilla-PCAT", leave=False):
            idx = torch.randint(n, (min(self.batch_size, n),), device=self.device)
            opt.zero_grad()
            self.criterion(F_s(f_tilde(self.X_aux[idx])), self.y_aux[idx]).backward()
            opt.step()
        for p in F_s.parameters():
            p.requires_grad_(True)
        return self._evaluate(self.F_c, F_s, f_tilde)

    def run_independent(self):
        set_seed(self.seed + 3)
        c, s = build_split_models(self.model_name, self.cut, self.dataset, device=self.device)
        opt = optim.Adam(list(c.parameters()) + list(s.parameters()), lr=self.lr)
        c.train(); s.train()
        n = len(self.y_aux)
        for it in tqdm(range(self.vanilla_iters), desc="  Independent", leave=False):
            idx = torch.randint(n, (min(self.batch_size, n),), device=self.device)
            opt.zero_grad()
            self.criterion(s(c(self.X_aux[idx])), self.y_aux[idx]).backward()
            opt.step()
        return evaluate_accuracy(c, s, make_loader(self.raw['test_plain'], batch_size=256), self.device)

    def run(self):
        last = self.run_pcat()
        v_acc, van_acc, van_mse = self.run_vanilla_pcat()
        ind_acc = self.run_independent()
        res = {
            'attack': 'model_stealing_pcat', 'defense': self.tag,
            'model': self.model_name, 'cut_layer': self.cut, 'dataset': self.dataset,
            'aux_records': len(self.aux_idx), 'aux_per_class': self.aux_per_class,
            'aux_pct_of_private': 100 * len(self.aux_idx) / len(self.private_idx),
            'start_batch': self.start_batch, 'epochs': self.epochs,
            'victim_acc': last['victim_acc'],
            'pcat_stolen_acc': last['stolen_acc'], 'pcat_gap': last['victim_acc'] - last['stolen_acc'],
            'pcat_fidelity_mse': last['fidelity_mse'],
            'vanilla_pcat_stolen_acc': van_acc, 'vanilla_pcat_gap': v_acc - van_acc,
            'vanilla_pcat_fidelity_mse': van_mse,
            'independent_acc': ind_acc,
        }
        print("\n" + "=" * 64)
        print("   PCAT RESULTS (test accuracy)")
        print("=" * 64)
        print(f"  Victim SL model        : {res['victim_acc']:.2f}%")
        print(f"  PCAT stolen model      : {res['pcat_stolen_acc']:.2f}%  (gap {res['pcat_gap']:.2f})")
        print(f"  Vanilla-PCAT           : {res['vanilla_pcat_stolen_acc']:.2f}%  (gap {res['vanilla_pcat_gap']:.2f})")
        print(f"  Independent (aux only) : {res['independent_acc']:.2f}%")
        print(f"  Fidelity MSE PCAT/Van. : {res['pcat_fidelity_mse']:.4f} / {res['vanilla_pcat_fidelity_mse']:.4f}")
        print("=" * 64)
        with open(os.path.join(self.cache_dir,
                               f"result_{self.model_name.lower()}_cut{self.cut}_{self.dataset}_{self.tag}.json"), 'w') as f:
            json.dump({'summary': res, 'history': self.history}, f, indent=2)
        return res