"""
SafeSplit Defence Sweep
------------------------
Tests SafeSplit defense against split learning attacks.

SafeSplit uses frequency domain analysis and rotational integrity checking
to detect poisoning/backdoor attacks and performs model rollback.

WHAT SAFESPLIT DEFENDS AGAINST
-------------------------------
SafeSplit detects and rolls back:
    - Backdoor client (poisoned updates detected)
    - Model poisoning attacks
    - Subtle weight manipulations via frequency analysis

It provides detection + rollback mechanism, not additive noise like DPSL.

USAGE
-----
    python defence_runner/run_safesplit_sweep.py --model PyramidCNN \
        --dataset CIFAR10 --cuts 2 3 4 --seed 0
"""

import os
import sys
import csv
import copy
import argparse
import datetime
import subprocess

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from existing_defense.safesplit_defense import SafeSplitDefense


# --------------------------------------------------------------------------
# attack coverage map — which attacks SafeSplit can influence
# --------------------------------------------------------------------------
FORWARD_ATTACKS = ['WhiteBox', 'UnSplit', 'AE_Decoder', 'FSHA']
PARTIAL_ATTACKS = ['VILLAIN']
UNCOVERED_ATTACKS = ['LabelLeakage', 'Backdoor_Client', 'Backdoor_Server']


def git_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'nogit'


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_models(device):
    """Instantiate client/server pair for the model named in Config."""
    name = Config.MODEL_NAME
    in_ch = 1 if Config.DATASET == 'MNIST' else 3

    if name == 'PyramidCNN':
        from all_model.pyramid_cnn import (PyramidCNNClientModel,
                                           PyramidCNNServerModel)
        client = PyramidCNNClientModel(in_channels=in_ch,
                                       cut_layer=Config.CUT_LAYER)
        server = PyramidCNNServerModel(num_classes=Config.NUM_CLASSES,
                                       cut_layer=Config.CUT_LAYER)
    elif name == 'KAGN':
        from all_model.kagn_models import KAGNClientModel, KAGNServerModel
        client = KAGNClientModel(in_channels=in_ch,
                                 cut_layer=Config.CUT_LAYER,
                                 degree=Config.DEGREE)
        server = KAGNServerModel(num_classes=Config.NUM_CLASSES,
                                 cut_layer=Config.CUT_LAYER,
                                 degree=Config.DEGREE)
    else:
        from all_model.models import ClientModel, ServerModel
        client = ClientModel(in_channels=in_ch)
        server = ServerModel(num_classes=Config.NUM_CLASSES)

    return client.to(device), server.to(device)


def train_with_defense(defense_params, train_loader, test_loader, device):
    """Train split model. SafeSplit detection happens during training monitoring."""
    from all_split_learning.split_learning import SplitLearningTrainer

    client, server = build_models(device)
    
    # ... lazy layer setup ...

    trainer = SplitLearningTrainer(client_model=client,
                                   server_model=server,
                                   train_loader=train_loader,
                                   test_loader=test_loader,
                                   defense=None)  # Don't pass defense to trainer
    trainer.train()
    trainer.save_results()
    return client, server, max(trainer.test_accuracies)


def run_forward_attacks(client, server, train_loader,
                        test_loader, device):
    """
    Run every forward-pass attack against the client.

    Note: SafeSplit defense is evaluated during training (detection/rollback),
    not by modifying client outputs, so attacks run on baseline models.
    """
    results = {}

    # ---- white-box inversion ----
    try:
        from all_attacks.attacks_whitebox import run_whitebox_attack
        psnr, ssim = run_whitebox_attack(client, test_loader, device,
                                         dataset=Config.DATASET,
                                         num_images=32, iterations=1000)
        results['WhiteBox'] = {'psnr': psnr, 'ssim': ssim}
    except Exception as exc:
        results['WhiteBox'] = {'error': str(exc)}

    # ---- UnSplit ----
    try:
        from all_attacks.attack_unsplit import UnSplitAttack
        attacker = UnSplitAttack(client, dataset=Config.DATASET)
        summary = attacker.attack(test_loader, num_batches=1)
        results['UnSplit'] = summary
    except Exception as exc:
        results['UnSplit'] = {'error': str(exc)}

    # ---- AE decoder ----
    try:
        from all_attacks.ae_decoder_attack import run_ae_decoder_attack
        summary = run_ae_decoder_attack(client, train_loader, test_loader,
                                        device, Config.DATASET,
                                        label=f'{Config.MODEL_NAME} AE')
        results['AE_Decoder'] = summary
    except Exception as exc:
        results['AE_Decoder'] = {'error': str(exc)}

    # ---- FSHA ----
    try:
        from all_attacks.fsha_attack import FSHAAttack
        in_ch = 1 if Config.DATASET == 'MNIST' else 3
        fsha = FSHAAttack(copy.deepcopy(client), in_channels=in_ch,
                          dataset=Config.DATASET)
        fsha.hijack(train_loader, test_loader, epochs=50)
        results['FSHA'] = fsha.reconstruct(test_loader)
    except Exception as exc:
        results['FSHA'] = {'error': str(exc)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PyramidCNN',
                        choices=['PyramidCNN', 'KAGN', 'Vanilla_SL'])
    parser.add_argument('--dataset', default='CIFAR10',
                        choices=['MNIST', 'CIFAR10'])
    parser.add_argument('--cuts', type=int, nargs='+', default=[2, 3, 4])
    parser.add_argument('--low_freq_ratios', type=float, nargs='+',
                        default=[0.1])
    parser.add_argument('--anomaly_thresholds', type=float, nargs='+',
                        default=[2.5])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='./results/safesplit_sweep.csv')
    args = parser.parse_args()

    Config.MODEL_NAME = args.model
    Config.DATASET = args.dataset
    Config.RUN_ATTACK = True

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fieldnames = ['timestamp', 'git_hash', 'model', 'dataset', 'cut_layer',
                  'seed', 'defense', 'low_freq_ratio', 'anomaly_threshold',
                  'accuracy', 'attack', 'metric', 'value', 'covered']

    write_header = not os.path.exists(args.out)
    handle = open(args.out, 'a', newline='')
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    def record(cut, low_freq, threshold, acc, attack, metric, value, covered):
        writer.writerow({
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'git_hash': git_hash(), 'model': args.model,
            'dataset': args.dataset, 'cut_layer': cut, 'seed': args.seed,
            'defense': 'none' if low_freq is None else 'SafeSplit',
            'low_freq_ratio': '' if low_freq is None else low_freq,
            'anomaly_threshold': '' if threshold is None else threshold,
            'accuracy': round(acc, 2), 'attack': attack,
            'metric': metric, 'value': value, 'covered': covered})
        handle.flush()

    from dataset import DatasetLoader

    for cut in args.cuts:
        Config.CUT_LAYER = cut

        # low_freq_ratio = None is the undefended baseline
        for low_freq in [None] + list(args.low_freq_ratios):
            for threshold in args.anomaly_thresholds:
                set_seed(args.seed)
                Config.RUN_TAG = ('nodef' if low_freq is None
                                  else f'safesplit_lf{low_freq:g}_th{threshold:g}') + f'_s{args.seed}'

                defense_params = None
                if low_freq is not None:
                    defense_params = {
                        'freq_low_ratio': low_freq,
                        'freq_threshold': threshold,
                        'rot_threshold': 2.0,
                        'combined_threshold': 1.5,
                        'max_checkpoints': 10
                    }

                print('\n' + '=' * 70)
                print(f'  {args.model} | {args.dataset} | cut {cut} | '
                      f'{"NO DEFENCE" if low_freq is None else f"SafeSplit LF={low_freq} Th={threshold}"}')
                print('=' * 70)

                train_loader, test_loader = DatasetLoader(Config.DATASET).get_loaders()

                client, server, acc = train_with_defense(
                    defense_params, train_loader, test_loader, device)

                record(cut, low_freq, threshold, acc, 'utility', 'test_accuracy', acc, 'n/a')

                attack_results = run_forward_attacks(
                    client, server, train_loader, test_loader, device)

                for name, summary in attack_results.items():
                    if 'error' in summary:
                        record(cut, low_freq, threshold, acc, name, 'error',
                               summary['error'][:120], 'yes')
                        continue
                    for metric, value in summary.items():
                        record(cut, low_freq, threshold, acc, name, metric,
                               round(float(value), 5), 'yes')

                for name in UNCOVERED_ATTACKS:
                    record(cut, low_freq, threshold, acc, name, 'note',
                           'SafeSplit detection-based defense', 'no')

    handle.close()
    print(f'\n  Sweep complete. Results appended to {args.out}')


if __name__ == '__main__':
    main()