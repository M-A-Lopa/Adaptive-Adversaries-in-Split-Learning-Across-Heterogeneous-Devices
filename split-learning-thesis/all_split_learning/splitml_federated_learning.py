"""
SplitML training (Alg. 4) and consensus inference (Alg. 5) of Trivedi et al. 2026.
Every round: each client trains its whole model M_k for one epoch on its private shard (paper Sec. 4.2),
then the shared top layers are averaged under multi-key CKKS. No smashed data or gradient is ever sent
to the server during training.
"""
import time
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import Config
from all_model.models import ClientModel
from all_model.kagn_models import KAGNClientModel
from all_model.pyramid_cnn import PyramidCNNClientModel
from all_defences.splitml_defense import SplitMLFederation, SplitMLConsensus, low_confidence_mask


# ============================================================================ SplitML client models
# M_k = bottom_k(top(x)): shared top = your thesis client model, private bottoms differ per client.


def build_shared_top(in_channels):
    if Config.MODEL_NAME == "KAGN":
        return KAGNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels, degree=Config.DEGREE)
    if Config.MODEL_NAME == "PyramidCNN":
        return PyramidCNNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_channels)
    return ClientModel(in_channels=in_channels)


def build_private_bottom(variant, num_classes):
    """variant 0 = your ServerModel head, 1 = deeper head, 2 = shallow head (no hidden layer, like paper Model-2/3)."""
    v = variant % 3
    if v == 0:
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(), nn.Linear(64 * 4 * 4, 256), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(256, num_classes))
    if v == 1:
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(), nn.Linear(128 * 2 * 2, 128), nn.ReLU(), nn.Linear(128, num_classes))
    return nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.LazyLinear(num_classes))


class SplitMLClientModel(nn.Module):
    def __init__(self, in_channels, num_classes, variant):
        super().__init__()
        self.variant = variant
        self.top = build_shared_top(in_channels)
        self.bottom = build_private_bottom(variant, num_classes)

    def forward(self, x):
        return self.bottom(self.top(x))


def build_splitml_clients(num_clients, in_channels, num_classes, device):
    img = 28 if Config.DATASET == 'MNIST' else 32
    clients = []
    for k in range(num_clients):
        m = SplitMLClientModel(in_channels, num_classes, variant=k)
        with torch.no_grad():
            m(torch.zeros(2, in_channels, img, img))          # materialise Lazy layers
        clients.append(m.to(device))
    return clients


def split_iid(dataset, num_clients, seed=0):
    n = len(dataset)
    sizes = [n // num_clients] * num_clients
    sizes[-1] += n - sum(sizes)
    return random_split(dataset, sizes, generator=torch.Generator().manual_seed(seed))


class SplitMLTrainer:
    def __init__(self, clients, train_dataset, test_loader, rounds, collaborative=True,
                 extra_dp_sigma=0.0, tag="splitml"):
        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.clients = clients
        self.K = len(clients)
        self.loaders = [DataLoader(s, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
                        for s in split_iid(train_dataset, self.K)]
        self.test_loader = test_loader
        self.rounds = rounds
        self.collaborative = collaborative
        self.optimizers = [torch.optim.Adam(c.parameters(), lr=Config.LEARNING_RATE) for c in clients]
        self.criterion = nn.CrossEntropyLoss()
        self.federation = SplitMLFederation(self.K, extra_dp_sigma=extra_dp_sigma) if collaborative else None
        self.history = []
        self.fed_time = 0.0
        self.tag = tag if collaborative else f"{tag}_standalone"

    def _local_epoch(self, k, r):
        model, opt = self.clients[k], self.optimizers[k]
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in tqdm(self.loaders[k], desc=f"  Round {r+1} client {k+1}", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad()
            out = model(x)
            loss = self.criterion(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)
        return loss_sum / len(self.loaders[k]), 100. * correct / total

    @torch.no_grad()
    def evaluate_client(self, k):
        m = self.clients[k]
        m.eval()
        correct = total = 0
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            correct += m(x).argmax(1).eq(y).sum().item()
            total += y.size(0)
        return 100. * correct / total

    def train(self):
        mode = "SplitML (encrypted FedAvg of shared top)" if self.collaborative else "Standalone (beta = 0)"
        print("\n" + "=" * 60)
        print(f"   {mode}")
        print("=" * 60)
        if self.collaborative:
            print(f"  CKKS backend : {self.federation.backend.name}")
        print(f"  Clients: {self.K} | Rounds: {self.rounds} | Heads: {[c.variant for c in self.clients]}")
        for r in range(self.rounds):
            stats = [self._local_epoch(k, r) for k in range(self.K)]
            if self.collaborative:
                t0 = time.time()
                self.federation.aggregate([c.top for c in self.clients])
                self.fed_time += time.time() - t0
            accs = [self.evaluate_client(k) for k in range(self.K)]
            row = {'round': r + 1, 'federation_time_s': self.fed_time}
            for k in range(self.K):
                row[f'client{k+1}_train_loss'], row[f'client{k+1}_train_acc'] = stats[k]
                row[f'client{k+1}_test_acc'] = accs[k]
            self.history.append(row)
            print(f"  Round {r+1:3d}/{self.rounds} | " + " | ".join(f"C{k+1}: {a:.2f}%" for k, a in enumerate(accs))
                  + (f" | fed {self.fed_time:.1f}s" if self.collaborative else ""))
        self._save()
        return self.history[-1]

    @torch.no_grad()
    def evaluate_consensus(self, lam=0.10, max_batches=None):
        """For every requester: accuracy of its own model vs TL / TP consensus on its low-confidence samples."""
        consensus = SplitMLConsensus()
        results = []
        for m in self.clients:
            m.eval()
        for j in range(self.K):
            req, peers = self.clients[j], [c for i, c in enumerate(self.clients) if i != j]
            own_all = tl_all = tp_all = total = n_q = own_q = tl_q = tp_q = 0
            for b, (x, y) in enumerate(self.test_loader):
                if max_batches and b >= max_batches:
                    break
                x, y = x.to(self.device), y.to(self.device)
                probs = torch.softmax(req(x), dim=1)
                own = probs.argmax(1)
                tl, tp = own.clone(), own.clone()
                mask = low_confidence_mask(probs, lam)
                if mask.any():
                    tl[mask] = consensus.query(req, peers, x[mask], mode="TL")
                    tp[mask] = consensus.query(req, peers, x[mask], mode="TP")
                    n_q += int(mask.sum())
                    own_q += own[mask].eq(y[mask]).sum().item()
                    tl_q += tl[mask].eq(y[mask]).sum().item()
                    tp_q += tp[mask].eq(y[mask]).sum().item()
                own_all += own.eq(y).sum().item()
                tl_all += tl.eq(y).sum().item()
                tp_all += tp.eq(y).sum().item()
                total += y.size(0)
            results.append({'requester': j + 1, 'queried_samples': n_q,
                            'own_acc': 100. * own_all / total, 'TL_acc': 100. * tl_all / total,
                            'TP_acc': 100. * tp_all / total,
                            'own_acc_on_queried': 100. * own_q / max(n_q, 1),
                            'TL_acc_on_queried': 100. * tl_q / max(n_q, 1),
                            'TP_acc_on_queried': 100. * tp_q / max(n_q, 1)})
            r = results[-1]
            print(f"  Requester C{j+1}: queried {n_q:5d} | own {r['own_acc']:.2f}% | TL {r['TL_acc']:.2f}% | "
                  f"TP {r['TP_acc']:.2f}%   (on queried: {r['own_acc_on_queried']:.1f} / "
                  f"{r['TL_acc_on_queried']:.1f} / {r['TP_acc_on_queried']:.1f})")
        return results

    def _save(self):
        pd.DataFrame(self.history).to_csv(
            f"{Config.RESULTS_DIR}/{self.tag}_{Config.MODEL_NAME.lower()}_training_{Config.DATASET}.csv", index=False)
        torch.save({'clients': [c.state_dict() for c in self.clients], 'variants': [c.variant for c in self.clients],
                    'dataset': Config.DATASET},
                   f"{Config.SAVE_DIR}/{self.tag}_{Config.MODEL_NAME.lower()}_{Config.DATASET}.pth")
