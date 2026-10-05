import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config
from dataset import DatasetLoader
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel

SETTINGS = {
    'def_epochs': 3, 'attack_epochs': 5, 'ae_epochs': 50, 'fsha_epochs': 5, 'wb_iters': 1000,
    'max_images': 32, 'leak_epochs': 5, 'vanilla_epochs': 30, 'max_batches': 0,
    'poison_rate': 0.05, 'poison_target': 0, 'poison_patch': 4, 'poison_trigger': 1.0,
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup(model, dataset, cut, seed):
    Config.MODEL_NAME, Config.DATASET, Config.CUT_LAYER, Config.NUM_CLASSES = model, dataset, cut, 10
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    seed_all(seed)
    return torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')


def in_ch():
    return 1 if Config.DATASET == 'MNIST' else 3


def img_size():
    return 28 if Config.DATASET == 'MNIST' else 32


def build_client(device=None):
    if Config.MODEL_NAME == 'KAGN':
        m = KAGNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_ch(), degree=Config.DEGREE)
    elif Config.MODEL_NAME == 'PyramidCNN':
        m = PyramidCNNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_ch())
    else:
        m = ClientModel(in_channels=in_ch())
    return m.to(device) if device is not None else m


def build_server(num_classes=10, device=None):
    if Config.MODEL_NAME == 'KAGN':
        m = KAGNServerModel(cut_layer=Config.CUT_LAYER, num_classes=num_classes, in_channels=in_ch(), degree=Config.DEGREE)
    elif Config.MODEL_NAME == 'PyramidCNN':
        m = PyramidCNNServerModel(cut_layer=Config.CUT_LAYER, num_classes=num_classes, in_channels=in_ch())
    else:
        m = ServerModel(num_classes=num_classes)
    return m.to(device) if device is not None else m


def materialize(client, server, device):
    was = (client.training, server.training)
    client.eval()
    server.eval()
    with torch.no_grad():
        server(client(torch.zeros(2, in_ch(), img_size(), img_size(), device=device)))
    client.train(was[0])
    server.train(was[1])


def n_features(client, device):
    with torch.no_grad():
        return client(torch.zeros(1, in_ch(), img_size(), img_size(), device=device)).numel()


class Limited:
    def __init__(self, loader, max_batches):
        self.loader, self.max_batches = loader, max_batches

    def __iter__(self):
        for i, b in enumerate(self.loader):
            if self.max_batches and i >= self.max_batches:
                break
            yield b

    def __len__(self):
        n = len(self.loader)
        return min(n, self.max_batches) if self.max_batches else n

    def __getattr__(self, name):
        return getattr(self.loader, name)


def loaders():
    tr, te = DatasetLoader(dataset_name=Config.DATASET).get_loaders()
    mb = SETTINGS['max_batches']
    return (Limited(tr, mb), Limited(te, mb)) if mb else (tr, te)


def vanilla_path():
    return f"{Config.SAVE_DIR}/best_{Config.MODEL_NAME.lower()}_sl_{Config.DATASET}.pth"


@torch.no_grad()
def accuracy(client, server, loader, device, view=None):
    client.eval()
    server.eval()
    c = t = 0
    for b in loader:
        x, y = b[0].to(device), b[1].to(device)
        HOLDER.set(y)
        z = client(x)
        if view is not None:
            z = view(z)
        c += server(z).argmax(1).eq(y).sum().item()
        t += y.size(0)
    client.train()
    server.train()
    return 100.0 * c / max(t, 1)


def train_vanilla(device):
    path = vanilla_path()
    if os.path.exists(path):
        return path
    print(f"[ablation] no clean checkpoint at {path}; training vanilla SL for {SETTINGS['vanilla_epochs']} epochs")
    tr, te = loaders()
    c, s = build_client(device), build_server(10, device)
    materialize(c, s, device)
    co = torch.optim.Adam(c.parameters(), lr=Config.LEARNING_RATE)
    so = torch.optim.Adam(s.parameters(), lr=Config.LEARNING_RATE)
    crit = nn.CrossEntropyLoss()
    best = -1.0
    for ep in range(SETTINGS['vanilla_epochs']):
        c.train()
        s.train()
        for x, y in tqdm(tr, desc=f"  vanilla {ep + 1}/{SETTINGS['vanilla_epochs']}", leave=False):
            x, y = x.to(device), y.to(device)
            co.zero_grad()
            z = c(x)
            zl = z.detach().requires_grad_(True)
            so.zero_grad()
            loss = crit(s(zl), y)
            loss.backward()
            so.step()
            z.backward(zl.grad)
            co.step()
        acc = accuracy(c, s, te, device)
        print(f"  vanilla epoch {ep + 1}: test {acc:.2f}%")
        if acc > best:
            best = acc
            torch.save({'epoch': ep, 'client_state': c.state_dict(), 'server_state': s.state_dict(),
                        'best_acc': best, 'dataset': Config.DATASET, 'cut_layer': Config.CUT_LAYER,
                        'model_type': Config.MODEL_NAME, 'lr': Config.LEARNING_RATE}, path)
    return path


def load_vanilla(device):
    path = train_vanilla(device)
    ck = torch.load(path, map_location=device)
    c, s = build_client(device), build_server(10, device)
    materialize(c, s, device)
    c.load_state_dict(ck['client_state'])
    s.load_state_dict(ck['server_state'])
    return c, s, float(ck.get('best_acc', float('nan')))


class LabelHolder:
    def __init__(self):
        self.labels = None

    def set(self, y):
        self.labels = y.detach() if torch.is_tensor(y) else torch.as_tensor(y)

    def get(self, n, device):
        if self.labels is None:
            return None
        return self.labels[:n].to(device)


HOLDER = LabelHolder()


class Tapped:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for b in self.loader:
            HOLDER.set(b[1])
            yield b

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)


def labelled(view):
    def fn(z):
        return view(z, HOLDER.get(z.shape[0], z.device))
    return fn


def atk_whitebox(client, view, test_loader, device):
    from defence_runner.run_splitfss_evaluation import run_whitebox_vs_view
    s = run_whitebox_vs_view(client, labelled(view), Tapped(test_loader), device, Config.DATASET,
                             SETTINGS['max_images'], SETTINGS['wb_iters'])
    return {'ssim': float(s['mean_ssim']), 'psnr': float(s['mean_psnr'])}


def atk_ae(client, view, train_loader, test_loader, device):
    from defence_runner.run_splitfss_evaluation import DefendedClient
    from all_attacks.ae_decoder_attack import run_ae_decoder_attack
    s, _, _ = run_ae_decoder_attack(DefendedClient(client, labelled(view)), Tapped(train_loader), Tapped(test_loader),
                                    device, Config.DATASET, ae_epochs=SETTINGS['ae_epochs'], label='ablation AE')
    return {'ssim': float(s['mean_ssim']), 'psnr': float(s['mean_psnr'])}


def atk_fsha(view, train_loader, test_loader, tag):
    from defence_runner.run_splitfss_evaluation import run_fsha_vs_view
    s = run_fsha_vs_view(labelled(view), in_ch(), Tapped(train_loader), test_loader, tag=tag,
                         epochs=SETTINGS['fsha_epochs'], num_images=SETTINGS['max_images'],
                         client_builder=build_client)
    return {'ssim': float(s['ssim']), 'psnr': float(s['psnr'])}


def atk_label_leakage(server_wrap, train_dataset, device):
    from all_attacks.label_leakage_attack import GradientNormLabelLeakageAttack, build_binary_split_loaders
    seed_all(0)
    btr, bte = build_binary_split_loaders(train_dataset, target_class=0, positive_ratio=0.1, batch_size=128, seed=0)
    c, s = build_client(device), build_server(1, device)
    materialize(c, s, device)
    atk = GradientNormLabelLeakageAttack(client_model=c, server_model=s, dataset=Config.DATASET,
                                         target_class=0, learning_rate=1e-4)
    if server_wrap is not None:
        atk.server_model = server_wrap(atk.server_model, c)
    summ = atk.run(Tapped(btr), bte, epochs=SETTINGS['leak_epochs'])
    atk.client_model.eval()
    atk.server_model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in bte:
            logits = atk.server_model(atk.client_model(x.to(device))).squeeze(1)
            correct += ((logits.cpu() > 0).float() == y.float()).sum().item()
            total += len(y)
    auc = float(np.nanmax([summ['q95_norm_leak_auc_cut'], summ['q95_cosine_leak_auc_cut']]))
    return {'auc': auc, 'acc': 100.0 * correct / max(total, 1)}


def atk_poison_client(server_wrap, train_loader, test_loader, device, tag):
    from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack
    seed_all(0)
    train_vanilla(device)
    c, s = build_client(device), build_server(10, device)
    materialize(c, s, device)
    atk = BackdoorPoisonAttack(client_model=c, server_model=s, base_dataset=train_loader.dataset,
                               dataset=Config.DATASET, num_classes=10, mode='client',
                               target_label=SETTINGS['poison_target'], poison_rate=SETTINGS['poison_rate'],
                               patch_size=SETTINGS['poison_patch'], trigger_value=SETTINGS['poison_trigger'],
                               model_tag=tag)
    atk._save_checkpoint = lambda *a, **k: None
    atk.load_clean_init(vanilla_path())
    original = atk._poison_batch

    def poison_and_tap(x, y):
        out = original(x, y)
        HOLDER.set(out[1])
        return out

    atk._poison_batch = poison_and_tap
    if server_wrap is not None:
        atk.server_model = server_wrap(atk.server_model, atk)
    atk.train(Tapped(train_loader), test_loader, epochs=SETTINGS['attack_epochs'])
    cda, asr = atk.evaluate(Tapped(test_loader))
    return {'asr': float(asr), 'acc': float(cda)}


class DropServer(nn.Module):
    def __init__(self, server, reject_fn, refit_fn=None, eval_view=None):
        super().__init__()
        self.server = server
        self.reject_fn = reject_fn
        self.refit_fn = refit_fn
        self.eval_view = eval_view
        self.seen = 0
        self.rejected = 0

    def forward(self, z):
        training = torch.is_grad_enabled() and z.requires_grad
        if not training:
            if self.eval_view is not None:
                z = self.eval_view(z, HOLDER.get(z.shape[0], z.device))
            return self.server(z)
        mask = self.reject_fn(z.detach(), HOLDER.get(z.shape[0], z.device))
        out = self.server(z)
        self.seen += z.shape[0]
        if mask is None or not bool(mask.any()):
            return out
        mask = mask.to(out.device)
        self.rejected += int(mask.sum())
        return torch.where(mask.view(-1, *([1] * (out.dim() - 1))), out.detach(), out)


class ReplaceServer(nn.Module):
    def __init__(self, server, view, eval_view=None):
        super().__init__()
        self.server = server
        self.view = view
        self.eval_view = eval_view

    def forward(self, z):
        training = torch.is_grad_enabled() and z.requires_grad
        fn = self.view if training else self.eval_view
        if fn is not None:
            z = z + (fn(z, HOLDER.get(z.shape[0], z.device)) - z).detach()
        return self.server(z)


GATES = {'ssim': 0.15, 'psnr': 3.0, 'auc': 0.10, 'asr': 20.0}
METRIC_OF = {'whitebox': 'ssim', 'ae_decoder': 'ssim', 'fsha': 'ssim', 'unsplit': 'ssim',
             'label_leakage': 'auc', 'poison_client': 'asr', 'villain': 'asr'}


def protection(attack, full, off):
    m = METRIC_OF[attack]
    if full is None or off is None or m not in full or m not in off:
        return None
    return off[m] - full[m]


def substantial(attack, full, off):
    g = protection(attack, full, off)
    return g is not None and g >= GATES[METRIC_OF[attack]]


def responsibility(attack, full, off, variant):
    m = METRIC_OF[attack]
    gap = protection(attack, full, off)
    if gap is None or gap <= 0 or variant is None or m not in variant:
        return None
    return (variant[m] - full[m]) / gap
