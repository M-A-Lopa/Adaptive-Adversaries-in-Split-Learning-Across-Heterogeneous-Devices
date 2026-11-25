import os
import sys
import time
import random
import argparse
import traceback

_r = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_r, 'config.py')) and os.path.dirname(_r) != _r:
    _r = os.path.dirname(_r)
sys.path.insert(0, _r)
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
from existing_defense.zorro_defense import ZORRODefense, ZorroEntry, cpu_state

BLANK = "—"
DEFENSE_NAME = "zorro"
EXPECTED_ATTACKS = ['WhiteBox', 'UnSplit', 'AE_Decoder', 'FSHA', 'LabelLeakage',
                    'VILLAIN', 'BackdoorPoison_Client', 'BackdoorPoison_Server']
DEFENSE_ACTIVE = {'WhiteBox': 'no (no hook: ZORRO does not alter smashed data)',
                  'UnSplit': 'no (no hook)', 'AE_Decoder': 'no (no hook)', 'FSHA': 'no (no hook)',
                  'LabelLeakage': 'no (no hook)', 'VILLAIN': 'yes (injection phase, one turn per epoch)',
                  'BackdoorPoison_Client': 'yes (one turn per epoch)',
                  'BackdoorPoison_Server': 'no (attacker is the server; outside ZORRO threat model)'}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_checked(module, state, what):
    missing, unexpected = module.load_state_dict(state, strict=False)
    assert not missing, f"[{what}] checkpoint is missing keys: {list(missing)[:5]}"
    assert not unexpected, f"[{what}] checkpoint has unexpected keys: {list(unexpected)[:5]}"


def materialize_lazy(client, server, device, in_channels=3, image_size=32):
    was = (client.training, server.training)
    client.eval(); server.eval()
    with torch.no_grad():
        server(client(torch.zeros(2, in_channels, image_size, image_size, device=device)))
    client.train(was[0]); server.train(was[1])


def make_defense(args, device):
    return ZORRODefense(k=args.k, beta=args.beta, transform=args.transform, band=args.band,
                        metric=args.metric, rollback=not args.no_rollback, commit=not args.no_commit,
                        zkp=not args.no_zkp, inspect=args.inspect, device=str(device))


def load_bm(defense, client, server):
    bm = defense.current_bm()
    client.load_state_dict(bm.client_state)
    server.load_state_dict(bm.server_state)


def zorro_turn(defense, turn, client_id, client, server, run_epoch, poisoned, cheat=False):
    load_bm(defense, client, server)
    before = defense.inspected_vector(client, server)
    run_epoch()
    after = defense.inspected_vector(client, server)
    entry = ZorroEntry(turn=turn, client_state=cpu_state(client), server_state=cpu_state(server),
                       update=after - before, poisoned=poisoned)
    rec = defense.process_turn(turn, client_id, entry, cheat=cheat)
    load_bm(defense, client, server)
    return rec


def seed_queue(defense, client, server, benign_epoch):
    for i in range(defense.k):
        before = defense.inspected_vector(client, server)
        benign_epoch()
        after = defense.inspected_vector(client, server)
        defense.seed(ZorroEntry(turn=-(defense.k - i), client_state=cpu_state(client),
                                server_state=cpu_state(server), update=after - before, poisoned=False))
    load_bm(defense, client, server)


def make_schedule(n_malicious, pmr, seed):
    if pmr >= 1.0:
        return ['M'] * n_malicious
    total = int(round(n_malicious / pmr))
    rng = random.Random(seed)
    pos = set(rng.sample(range(total), n_malicious))
    return ['M' if i in pos else 'B' for i in range(total)]


def guarded_turns(atk, defense, schedule, malicious_epoch, benign_epoch, test_loader, label, cheat):
    hist = {'epoch': [], 'asr': [], 'cda': []}
    for t, kind in enumerate(schedule, 1):
        cid = 'attacker' if kind == 'M' else f'benign_{t}'
        if kind == 'M' and defense.is_banned('attacker'):
            defense.stats['skipped_banned'] += 1
            load_bm(defense, atk.client_model, atk.server_model)
            msg = "SKIPPED (attacker excluded after failed proof)"
        else:
            rec = zorro_turn(defense, t, cid, atk.client_model, atk.server_model,
                             malicious_epoch if kind == 'M' else benign_epoch,
                             poisoned=(kind == 'M'), cheat=(cheat and kind == 'M'))
            msg = (f"{'ACCEPTED' if rec['accepted'] else 'PROOF REJECTED'} | removed turn {rec['removed_turn']}"
                   f"{' (poisoned)' if rec['removed_poisoned'] else ''} | BM turn {rec['bm_turn']}"
                   f"{' (poisoned)' if rec['bm_poisoned'] else ''}")
        cda, asr = atk.evaluate(test_loader)
        hist['epoch'].append(t); hist['asr'].append(asr); hist['cda'].append(cda)
        print(f"  [ZORRO] {label} turn {t:2d}/{len(schedule)} ({kind}) {msg} | ASR {asr:.2f}% | CDA {cda:.2f}%")
    atk.history = hist
    return defense.summary()


class ZorroTrainer:
    def __init__(self, client, server, train_loader, test_loader, device, ckpt_path, args):
        self.client, self.server, self.device, self.ckpt_path = client, server, device, ckpt_path
        self.train_loader, self.test_loader = train_loader, test_loader
        self.client_opt = torch.optim.Adam(client.parameters(), lr=Config.LEARNING_RATE)
        self.server_opt = torch.optim.Adam(server.parameters(), lr=Config.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
        self.defense = make_defense(args, device)
        self.losses, self.train_accs, self.test_accs = [], [], []

    def _epoch(self):
        self.client.train(); self.server.train()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(self.train_loader, desc="  ZORRO benign turn", leave=False):
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
        self._last = (run_loss / len(self.train_loader), 100.0 * correct / total)

    @torch.no_grad()
    def evaluate(self):
        self.client.eval(); self.server.eval()
        c = t = 0
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            c += self.server(self.client(x)).argmax(1).eq(y).sum().item(); t += y.size(0)
        return 100.0 * c / t

    def train(self):
        print("\n" + "=" * 60 + f"\n   ZORRO SPLIT LEARNING — {Config.MODEL_NAME} | cut {Config.CUT_LAYER}\n" + "=" * 60)
        k = self.defense.k
        best = 0.0
        for epoch in range(1, Config.EPOCHS + 1):
            if epoch <= k:
                before = self.defense.inspected_vector(self.client, self.server)
                self._epoch()
                after = self.defense.inspected_vector(self.client, self.server)
                self.defense.seed(ZorroEntry(turn=epoch, client_state=cpu_state(self.client),
                                             server_state=cpu_state(self.server), update=after - before,
                                             poisoned=False))
                rec = {'removed_turn': None, 'bm_turn': epoch}
            else:
                rec = zorro_turn(self.defense, epoch, 'benign_client', self.client, self.server, self._epoch,
                                 poisoned=False)
            te = self.evaluate()
            loss, tr = self._last
            self.losses.append(loss); self.train_accs.append(tr); self.test_accs.append(te)
            print(f"  Epoch {epoch:3d}/{Config.EPOCHS} | Loss {loss:.4f} | Train {tr:.2f}% | Test(BM) {te:.2f}% | "
                  f"removed turn {rec['removed_turn']} | BM turn {rec['bm_turn']}")
            if te > best:
                best = te
                torch.save({'epoch': epoch, 'client_state': self.client.state_dict(),
                            'server_state': self.server.state_dict(), 'best_acc': best,
                            'dataset': Config.DATASET, 'model_type': Config.MODEL_NAME,
                            'cut_layer': Config.CUT_LAYER, 'defense': DEFENSE_NAME,
                            'zorro_config': self.defense.config()}, self.ckpt_path)
        print(f"  ZORRO benign training done. Best test accuracy {best:.2f}% | {self.defense.summary()}")
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
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--beta', type=float, default=0.7)
    ap.add_argument('--transform', default='dct', choices=['dct', 'none'])
    ap.add_argument('--band', default='low', choices=['low', 'all', 'high'])
    ap.add_argument('--metric', default='l1', choices=['l1', 'l2', 'cosine'])
    ap.add_argument('--no-rollback', action='store_true')
    ap.add_argument('--no-commit', action='store_true')
    ap.add_argument('--no-zkp', action='store_true')
    ap.add_argument('--inspect', default='client', choices=['client', 'client+server'])
    ap.add_argument('--cheat', action='store_true', help="attacker does not run Algorithm 1 honestly")
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
    print(f"Device: {device} | Defense: ZORRO ({args.variant}) | {make_defense(args, 'cpu')}")

    train_loader, test_loader = DatasetLoader(dataset_name=Config.DATASET).get_loaders()
    in_channels = 3
    checkpoint_path = f"{Config.SAVE_DIR}/best_{DEFENSE_NAME}_paper_{TAG}.pth"

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
        assert ck.get('defense') == DEFENSE_NAME, "Checkpoint is not a ZORRO checkpoint"
        if Config.MODEL_NAME != "Vanilla_SL":
            assert ck['cut_layer'] == Config.CUT_LAYER
        return ck

    def build_fresh_split(num_classes):
        c, s = build_client(), build_server(num_classes)
        ck = _load_ckpt()
        load_checked(c, ck['client_state'], 'zorro client')
        if num_classes == Config.NUM_CLASSES:
            load_checked(s, ck['server_state'], 'zorro server')
        else:
            materialize_lazy(c, s, device, in_channels)
        return c, s

    client_model, server_model = build_client(), build_server(Config.NUM_CLASSES)
    materialize_lazy(client_model, server_model, device, in_channels)

    if os.path.exists(checkpoint_path):
        print(f"\n[✓] Found ZORRO checkpoint: {checkpoint_path} — skipping training")
    else:
        print(f"\n[!] No checkpoint at {checkpoint_path} — training with ZORRO (paper defaults)")
        default_args = ap.parse_args([])
        trainer = ZorroTrainer(client_model, server_model, train_loader, test_loader, device,
                               checkpoint_path, default_args)
        losses, tr_accs, te_accs = trainer.train()
        pd.DataFrame({'epoch': range(1, len(losses) + 1), 'train_loss': losses, 'train_accuracy': tr_accs,
                      'test_accuracy': te_accs}).to_csv(f"{Config.RESULTS_DIR}/{DEFENSE_NAME}_results_{TAG}.csv",
                                                        index=False)
        pd.DataFrame([trainer.defense.summary()]).to_csv(f"{Config.RESULTS_DIR}/{DEFENSE_NAME}_stats_{TAG}.csv",
                                                          index=False)
        pd.DataFrame(trainer.defense.log).to_csv(f"{Config.RESULTS_DIR}/{DEFENSE_NAME}_turns_train_{TAG}.csv",
                                                 index=False)

    ck = _load_ckpt()
    load_checked(client_model, ck['client_state'], 'zorro client')
    load_checked(server_model, ck['server_state'], 'zorro server')
    client_model.eval(); server_model.eval()
    c = t = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            c += server_model(client_model(x)).argmax(1).eq(y).sum().item(); t += y.size(0)
    clean_acc = 100.0 * c / t
    print(f"\n  Clean test accuracy (ZORRO, {Config.MODEL_NAME}, cut {Config.CUT_LAYER}): {clean_acc:.2f}%")
    if args.skip_attacks:
        sys.exit(0)

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
        print("\n" + "=" * 60 + f"\n  {title} vs ZORRO[{args.variant}] — {Config.MODEL_NAME} (cut {Config.CUT_LAYER})\n" + "=" * 60)
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
        pd.DataFrame(defense.log).to_csv(f"{R}/{DEFENSE_NAME}_turns_{name.lower()}_{VTAG}.csv", index=False)

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
                                        label=f'{Config.MODEL_NAME} ZORRO AE Decoder')
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

        defense = make_defense(args, device)
        atk._set_client_learning_rate(atk.attack_learning_rate)
        seed_queue(defense, atk.client_model, atk.server_model, benign_epoch)
        stats = guarded_turns(atk, defense, make_schedule(INJECTION_EPOCHS, args.pmr, args.seed),
                              malicious_epoch=lambda: atk.inject_backdoor(idx_loader, test_loader, epochs=1),
                              benign_epoch=benign_epoch, test_loader=test_loader, label='VILLAIN', cheat=args.cheat)
        record_activity('VILLAIN', defense, stats)
        s = atk.summarise(test_loader=test_loader, clean_baseline=base)
        pd.DataFrame([{**s, **{f'zorro_{k}': v for k, v in stats.items()}}]).to_csv(
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
                print("  [ZORRO] NOT applicable: the attacker is the server. Running unmonitored.")
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

                defense = make_defense(args, device)
                seed_queue(defense, atk.client_model, atk.server_model, benign_epoch)
                stats = guarded_turns(atk, defense, make_schedule(BD_EPOCHS, args.pmr, args.seed),
                                      malicious_epoch=malicious_epoch, benign_epoch=benign_epoch,
                                      test_loader=test_loader, label='Backdoor(Client)', cheat=args.cheat)
                record_activity('BackdoorPoison_Client', defense, stats)
                extra = {f'zorro_{k}': v for k, v in stats.items()}
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
        v = results[a].get(METRICS[a][col])
        return "MISSING" if v is None else f"{float(v):.{dec}f}"

    W = 100
    print("\n" + "=" * W + f"\n    ZORRO[{args.variant}] — {Config.MODEL_NAME} | cut {Config.CUT_LAYER} | "
          f"clean acc {clean_acc:.2f}%\n" + "=" * W)
    print(f"{'Attack':<24}" + "".join(f"{h:>10}" for _, h, _ in COLS))
    for a in EXPECTED_ATTACKS:
        print(f"  {a:<22}" + "".join(f"{cell(a, c, d):>10}" for c, _, d in COLS))
    for name, st in activity.items():
        print(f"  ZORRO activity {name}: PRR {st['PRR']:.1f}% | BBR {st['BBR']:.1f}% | "
              f"proof rejected {st['proof_rejected']} | skipped (banned) {st['skipped_banned']}")

    rows = []
    for a in selected:
        row = {'defense': DEFENSE_NAME, 'variant': args.variant, 'model': Config.MODEL_NAME,
               'cut_layer': Config.CUT_LAYER, 'dataset': Config.DATASET, 'seed': args.seed, 'clean_acc': clean_acc,
               'attack': a, 'zorro_in_loop': DEFENSE_ACTIVE[a], 'cheat': args.cheat, 'pmr': args.pmr,
               'status': 'FAILED' if a in failures else 'ok', 'error': failures.get(a),
               **{f'cfg_{k}': v for k, v in make_defense(args, 'cpu').config().items()}}
        for c_, _, _ in COLS:
            row[c_] = None if (c_ not in METRICS[a] or a in failures) else results[a].get(METRICS[a][c_])
        if a in activity:
            row.update({f'zorro_{k}': v for k, v in activity[a].items() if not k.startswith('cfg_')})
        rows.append(row)
    out = f"{R}/{DEFENSE_NAME}_all_attacks_{VTAG}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Saved → {out}")