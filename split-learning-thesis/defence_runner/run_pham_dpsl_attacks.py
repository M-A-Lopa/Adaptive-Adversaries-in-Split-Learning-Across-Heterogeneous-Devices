

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback
import torch
import pandas as pd
from torch.utils.data import DataLoader

from config import Config
from dataset import DatasetLoader
from existing_defense.pham_dpsl_defense import (DPClientModel, ResizedDPClientModel, NoiseFreeView,
                                            build_base_split, build_full_server, dpsl_run_name,
                                            upgrade_client_state)
from all_split_learning.dpsl_split_learning import DPSLSplitLearningTrainer
from all_attacks.attacks_whitebox import WhiteBoxInversionAttack, AttackMetricsTracker
from all_attacks.attack_unsplit import UnSplitAttack
from all_attacks.ae_decoder_attack import run_ae_decoder_attack
from all_attacks.fsha_attack import FSHAAttack
from all_attacks.label_leakage_attack import build_binary_split_loaders
from all_attacks.villain_backdoor_attack import VILLAINBackdoorAttack, build_indexed_loader
from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack
from all_defences.label_protection_defense import LabelProtectionDefense, attach_label_protection
from defence_runner.run_label_protection import TrackedLabelLeakageAttack, GradientObserver


# ─────────────────────────── defence settings ───────────────────────────
EPSILON_VALUES = [None, 10, 5, 2]      # None = no noise (same pipeline, reference row)
DELTA          = 1e-5
NOISE_LOCATION = 'split'
USE_RESIZE     = False
TARGET_EPOCHS  = 100                   # training of the DP target model (paper: 100)
ATTACKER_KNOWS_DEFENSE = True

RUN = {'WhiteBox': True, 'UnSplit': True, 'AE_Decoder': True, 'FSHA': True,
       'LabelLeakage': True, 'VILLAIN': True, 'Backdoor_Client': True, 'Backdoor_Server': True}

# ─────────────────── attack settings (same as main.py) ──────────────────
MAX_IMAGES, ITERATIONS = 32, 1000
HIJACK_EPOCHS, CRITIC_ITERS = 50, 5
LEAKAGE_EPOCHS, POSITIVE_CLASS, POSITIVE_RATIO, LEAKAGE_BATCH, LEAKAGE_LR = 5, 0, 0.1, 128, 1e-4
WARMUP_EPOCHS, INFERENCE_EPOCHS, INJECTION_EPOCHS, VILLAIN_BATCH = 15, 5, 15, 32
TARGET_LABEL, TRIGGER_BETA, TRIGGER_FRACTION, DROPOUT_KEEP = 0, 1.0, 0.5, 0.75
GAMMA_LOW, GAMMA_HIGH, POISON_RATE, CANDIDATES = 0.6, 1.2, 0.05, 14
BACKDOOR_TARGET_LABEL, BACKDOOR_POISON_RATE, BACKDOOR_PATCH_SIZE = 0, 0.05, 4
BACKDOOR_TRIGGER_VALUE, BACKDOOR_TRAIN_EPOCHS, BACKDOOR_SURROGATE_EPOCHS = 1.0, 10, 5
# ─────────────────────────────────────────────────────────────────────────

IN_CH = 1 if Config.DATASET == 'MNIST' else 3
HW = 28 if Config.DATASET == 'MNIST' else 32
BASE_RESULTS = Config.RESULTS_DIR


def make_client(eps, noise=True):
    """Fresh DP client. noise=False -> same mechanism minus the random noise (clamp kept)."""
    base, _ = build_base_split(Config.MODEL_NAME, Config.CUT_LAYER, IN_CH, Config.NUM_CLASSES, Config.DEGREE)
    if USE_RESIZE:
        c = ResizedDPClientModel(base, (IN_CH, HW, HW), eps, DELTA, Config.DATASET)
    else:
        c = DPClientModel(base, eps, DELTA, NOISE_LOCATION, Config.DATASET)
    c.noise.active = noise
    return c


def make_server(num_classes):
    if USE_RESIZE:
        return build_full_server(Config.MODEL_NAME, Config.CUT_LAYER, IN_CH, num_classes, Config.DEGREE)
    return build_base_split(Config.MODEL_NAME, Config.CUT_LAYER, IN_CH, num_classes, Config.DEGREE)[1]


def attacker_client_builder(eps):
    """The attacker's own client-architecture model (UnSplit clone, FSHA pilot, surrogate)."""
    if ATTACKER_KNOWS_DEFENSE:
        return lambda: make_client(eps, noise=True)
    return lambda: build_base_split(Config.MODEL_NAME, Config.CUT_LAYER, IN_CH,
                                    Config.NUM_CLASSES, Config.DEGREE)[0]


def get_target_model(eps, train_loader, test_loader, device):
    """DP-trained client+server (shares checkpoints with run_pham_dpsl_single.py)."""
    name = dpsl_run_name(Config.CUT_LAYER, eps, NOISE_LOCATION, USE_RESIZE)
    client, server = make_client(eps).to(device), make_server(Config.NUM_CLASSES).to(device)
    ckpt = f"{Config.SAVE_DIR}/best_{Config.MODEL_NAME.lower()}_{name}_{Config.DATASET}.pth"
    trainer = DPSLSplitLearningTrainer(client, server, train_loader, test_loader, TARGET_EPOCHS, name)
    if os.path.exists(ckpt):
        print(f"  [✓] loading DP target model {ckpt}")
        st = torch.load(ckpt, map_location=device)
        client.load_state_dict(upgrade_client_state(st['client_state'])); server.load_state_dict(st['server_state'])
    else:
        print(f"  [!] no checkpoint {ckpt} -> training DP target model ({TARGET_EPOCHS} epochs)")
        trainer.train(); trainer.save_results()
    return client, server, trainer._evaluate()


def guarded(name, results, fn):
    if not RUN.get(name, False):
        return
    print("\n" + "=" * 70 + f"\n  {name}  —  {Config.MODEL_NAME} / {Config.DATASET} / {results['epsilon']}\n" + "=" * 70)
    try:
        results.update(fn())
    except Exception as e:                            # one failing attack must not kill the sweep
        traceback.print_exc()
        results[f'{name}_error'] = repr(e)


def run_all_attacks(eps, train_loader, test_loader, device):
    eps_name = 'no noise' if eps is None else f'eps={eps:g}'
    run_name = dpsl_run_name(Config.CUT_LAYER, eps, NOISE_LOCATION, USE_RESIZE)
    Config.RESULTS_DIR = os.path.join(BASE_RESULTS, 'pham_dpsl_attacks',
                                      f"{Config.MODEL_NAME.lower()}_{Config.DATASET}", run_name)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    tag = f"{Config.MODEL_NAME.lower()}_{Config.DATASET}_{run_name}"

    r = {'model': Config.MODEL_NAME, 'dataset': Config.DATASET, 'cut_layer': Config.CUT_LAYER,
         'variant': 'resize' if USE_RESIZE else NOISE_LOCATION, 'epsilon': eps_name,
         'attacker_knows_defense': ATTACKER_KNOWS_DEFENSE}

    target_c, target_s, acc = get_target_model(eps, train_loader, test_loader, device)
    r['sigma'] = round(target_c.sigma, 4)
    r['clean_task_acc'] = round(acc, 2)
    target_c.eval(); target_s.eval()

    # 1 ── White-box rMSE ───────────────────────────────────────────────
    def whitebox():
        mean, std = target_c.mean.to(device), target_c.std.to(device)
        attacker = WhiteBoxInversionAttack(client_model=NoiseFreeView(target_c), dataset=Config.DATASET,
                                           iterations=ITERATIONS, lr=1e-2)
        tracker, seen = AttackMetricsTracker(), 0
        for x, _ in test_loader:
            if seen >= MAX_IMAGES:
                break
            x = x[:MAX_IMAGES - seen].to(device)
            with torch.no_grad():
                target = target_c(x)                       # NOISY smashed data received by server
            rec = attacker.reconstruct(target, x.shape)
            tracker.log_batch(torch.clamp(x * std + mean, 0, 1), rec)
            seen += x.shape[0]
        s = tracker.get_summary()
        return {'wb_psnr': s['mean_psnr'], 'wb_ssim': s['mean_ssim']}
    guarded('WhiteBox', r, whitebox)

    # 2 ── UnSplit ──────────────────────────────────────────────────────
    def unsplit():
        atk = UnSplitAttack(client_model=target_c, in_channels=IN_CH,
                            clone_builder=attacker_client_builder(eps),
                            main_iters=getattr(Config, 'unsplit_main_iters', 100),
                            input_iters=getattr(Config, 'unsplit_input_iters', 20),
                            model_iters=getattr(Config, 'unsplit_model_iters', 20))
        s = atk.run_attack(test_loader, num_batches=max(MAX_IMAGES // Config.BATCH_SIZE, 1))
        return {'unsplit_psnr': s['psnr'], 'unsplit_ssim': s['ssim'], 'unsplit_mse': s['mse']}
    guarded('UnSplit', r, unsplit)

    # 3 ── AE decoder ───────────────────────────────────────────────────
    def ae():
        if USE_RESIZE:
            print("  NOTE: with resizing, the repo AE builder collapses to a tiny decoder for image-sized\n"
                  "        smashed data -> this attack is weakened; use run_pham_dpsl_single.py\n"
                  "        (inverse-network attack) for the leakage numbers of the resize variant.")
        s, _, _ = run_ae_decoder_attack(client_model=target_c, train_loader=train_loader,
                                        test_loader=test_loader, device=device, dataset=Config.DATASET,
                                        preprocess_fn=None, ae_epochs=50,
                                        label=f'{Config.MODEL_NAME} AE Decoder [DP-SL {eps_name}]')
        return {'ae_psnr': s['mean_psnr'], 'ae_ssim': s['mean_ssim'], 'ae_mse': s['mean_mse']}
    guarded('AE_Decoder', r, ae)

    # 4 ── FSHA ─────────────────────────────────────────────────────────
    def fsha():
        priv = DataLoader(train_loader.dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        pub = DataLoader(test_loader.dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        atk = FSHAAttack(client_model=make_client(eps).to(device), in_channels=IN_CH, dataset=Config.DATASET,
                         pilot_builder=attacker_client_builder(eps), critic_iters=CRITIC_ITERS)
        atk.hijack(priv, pub, epochs=HIJACK_EPOCHS)
        s = atk.reconstruct(train_loader, num_images=MAX_IMAGES)
        return {'fsha_psnr': s['psnr'], 'fsha_ssim': s['ssim'], 'fsha_mse': s['mse']}
    guarded('FSHA', r, fsha)

    # 5+6 ── Label leakage (norm + cosine) ──────────────────────────────
    def leakage():
        btr, bte = build_binary_split_loaders(train_loader.dataset, target_class=POSITIVE_CLASS,
                                              positive_ratio=POSITIVE_RATIO, batch_size=LEAKAGE_BATCH)
        atk = TrackedLabelLeakageAttack(client_model=make_client(eps), server_model=make_server(1),
                                        dataset=Config.DATASET, target_class=POSITIVE_CLASS,
                                        learning_rate=LEAKAGE_LR)
        obs = GradientObserver()
        attach_label_protection(atk, LabelProtectionDefense('no_noise'), obs)
        s = atk.run(btr, bte, epochs=LEAKAGE_EPOCHS)
        extra = obs.summary()
        out = {'leak_norm_auc_cut': s.get('q95_norm_leak_auc_cut'),
               'leak_cosine_auc_cut': s.get('q95_cosine_leak_auc_cut'),
               'leak_norm_auc_first': s.get('q95_norm_leak_auc_first'),
               'leak_cosine_auc_first': s.get('q95_cosine_leak_auc_first'),
               'leak_task_acc': atk.epoch_history['test_accuracy'][-1],
               'leak_chance_acc': 100 * extra['mean_chance_accuracy']}
        for a in ('norm', 'cosine'):
            out[f'leak_{a}_recovery_acc'] = 100 * extra[f'mean_{a}_recovery_acc']
            out[f'leak_{a}_recovery_bal_acc'] = 100 * extra[f'mean_{a}_recovery_bal_acc']
        atk.save_visualization(tag=tag)
        return out
    guarded('LabelLeakage', r, leakage)

    # 7 ── VILLAIN ──────────────────────────────────────────────────────
    def villain():
        idx = build_indexed_loader(train_loader.dataset, batch_size=VILLAIN_BATCH, shuffle=True)
        atk = VILLAINBackdoorAttack(client_model=make_client(eps), server_model=make_server(Config.NUM_CLASSES),
                                    base_dataset=train_loader.dataset, dataset=Config.DATASET,
                                    num_classes=Config.NUM_CLASSES, target_label=TARGET_LABEL, beta=TRIGGER_BETA,
                                    trigger_fraction=TRIGGER_FRACTION, dropout_keep=DROPOUT_KEEP,
                                    gamma_low=GAMMA_LOW, gamma_high=GAMMA_HIGH, poison_rate=POISON_RATE,
                                    candidates_per_batch=CANDIDATES)
        atk.warmup(idx, epochs=WARMUP_EPOCHS)
        base_acc, _ = atk.evaluate(test_loader)
        atk.infer_labels(idx, epochs=INFERENCE_EPOCHS)
        atk.fabricate_trigger(idx)
        atk.inject_backdoor(idx, test_loader, epochs=INJECTION_EPOCHS)
        s = atk.summarise(test_loader=test_loader, clean_baseline=base_acc)
        atk.save_visualization(tag=tag)
        return {'villain_lia': s['lia'], 'villain_asr': s['asr'], 'villain_cda': s['cda']}
    guarded('VILLAIN', r, villain)

    # 8a ── Backdoor, client-side ───────────────────────────────────────
    def bd_client():
        atk = BackdoorPoisonAttack(client_model=make_client(eps), server_model=make_server(Config.NUM_CLASSES),
                                   base_dataset=train_loader.dataset, dataset=Config.DATASET,
                                   num_classes=Config.NUM_CLASSES, mode='client', target_label=BACKDOOR_TARGET_LABEL,
                                   poison_rate=BACKDOOR_POISON_RATE, patch_size=BACKDOOR_PATCH_SIZE,
                                   trigger_value=BACKDOOR_TRIGGER_VALUE,
                                   model_tag=f"{Config.MODEL_NAME.lower()}_{run_name}")
        atk.train(train_loader, test_loader, epochs=BACKDOOR_TRAIN_EPOCHS)
        s = atk.summarise()
        atk.save_visualization(tag=tag)
        return {'bd_client_asr': s['asr'], 'bd_client_cda': s['cda']}
    guarded('Backdoor_Client', r, bd_client)

    # 8b ── Backdoor, server-side (surrogate client) ────────────────────
    def bd_server():
        atk = BackdoorPoisonAttack(client_model=make_client(eps), server_model=make_server(Config.NUM_CLASSES),
                                   base_dataset=train_loader.dataset, dataset=Config.DATASET,
                                   num_classes=Config.NUM_CLASSES, mode='server', target_label=BACKDOOR_TARGET_LABEL,
                                   poison_rate=BACKDOOR_POISON_RATE, patch_size=BACKDOOR_PATCH_SIZE,
                                   trigger_value=BACKDOOR_TRIGGER_VALUE,
                                   surrogate_builder=attacker_client_builder(eps),
                                   model_tag=f"{Config.MODEL_NAME.lower()}_{run_name}")
        atk.pretrain_server_backdoor(test_loader, epochs=BACKDOOR_SURROGATE_EPOCHS)
        atk.train(train_loader, test_loader, epochs=BACKDOOR_TRAIN_EPOCHS)
        s = atk.summarise()
        atk.save_visualization(tag=tag)
        return {'bd_server_asr': s['asr'], 'bd_server_cda': s['cda']}
    guarded('Backdoor_Server', r, bd_server)

    Config.RESULTS_DIR = BASE_RESULTS
    return r


def print_summary(df):
    cols = [('clean_task_acc', 'TaskAcc'), ('wb_ssim', 'WB SSIM'), ('unsplit_ssim', 'UnSp SSIM'),
            ('ae_ssim', 'AE SSIM'), ('fsha_ssim', 'FSHA SSIM'), ('leak_norm_auc_cut', 'NormAUC'),
            ('leak_cosine_auc_cut', 'CosAUC'), ('villain_asr', 'VIL ASR'),
            ('bd_client_asr', 'BDc ASR'), ('bd_server_asr', 'BDs ASR')]
    cols = [(c, h) for c, h in cols if c in df.columns]
    print("\n" + "=" * 110)
    print(f"  DP-SL (Pham et al.) vs ALL ATTACKS — {Config.MODEL_NAME} / {Config.DATASET} / cut {Config.CUT_LAYER}")
    print("  reconstruction: lower SSIM = better defence | label leakage: AUC -> 0.5 = better | backdoor: lower ASR = better")
    print("=" * 110)
    print(f"  {'epsilon':<10}" + "".join(f"{h:>10}" for _, h in cols))
    for _, row in df.iterrows():
        vals = []
        for c, _ in cols:
            v = row.get(c)
            if not isinstance(v, (int, float)) or v != v:          # missing / NaN / error
                vals.append(f"{'—':>10}")
            elif 'ssim' in c or 'auc' in c:
                vals.append(f"{v:>10.4f}")
            else:
                vals.append(f"{v:>10.2f}")
        print(f"  {row['epsilon']:<10}" + "".join(vals))
    print("=" * 110)


if __name__ == "__main__":
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    train_loader, test_loader = DatasetLoader(dataset_name=Config.DATASET).get_loaders()

    out_dir = os.path.join(BASE_RESULTS, 'pham_dpsl_attacks')
    os.makedirs(out_dir, exist_ok=True)
    variant = dpsl_run_name(Config.CUT_LAYER, None, NOISE_LOCATION, USE_RESIZE).rsplit('_', 1)[0]
    csv = f"{out_dir}/summary_{Config.MODEL_NAME.lower()}_{Config.DATASET}_{variant}.csv"

    rows = []
    for eps in EPSILON_VALUES:
        rows.append(run_all_attacks(eps, train_loader, test_loader, device))
        pd.DataFrame(rows).to_csv(csv, index=False)   # saved after every epsilon
    df = pd.DataFrame(rows)
    print_summary(df)
    print(f"\n  -> {csv}")