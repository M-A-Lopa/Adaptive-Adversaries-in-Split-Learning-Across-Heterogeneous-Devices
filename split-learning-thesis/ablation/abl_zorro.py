import os
import subprocess
import sys

import pandas as pd

from config import Config
import abl_common as C

ATTACK_NAME = {'poison_client': 'BackdoorPoison_Client', 'villain': 'VILLAIN'}


class ZorroAblation:
    name = 'zorro'
    attacks = ['poison_client', 'villain']

    def __init__(self):
        self.variants = {
            'full': [],
            'off': ['--no-rollback'],
            'C1_no_dct': ['--transform', 'none'],
            'C2_all_band': ['--band', 'all'],
            'C2_high_band_legacy': ['--band', 'high'],
            'C3_l2': ['--metric', 'l2'],
            'C3_cosine': ['--metric', 'cosine'],
            'C4_beta_1.0': ['--beta', '1.0'],
            'C4_beta_0.5': ['--beta', '0.5'],
            'C5_k1': ['--k', '1'],
            'C8_cheat_with_zkp': ['--cheat'],
            'C8_cheat_without_zkp': ['--cheat', '--no-zkp'],
            'X_inspect_client_and_server': ['--inspect', 'client+server'],
        }
        self.done = set()

    def _csv(self, variant):
        tag = f"{Config.MODEL_NAME.lower()}_cut{Config.CUT_LAYER}_{Config.DATASET}_{variant}"
        return f"{Config.RESULTS_DIR}/zorro_all_attacks_{tag}.csv"

    def _check_runner(self, script):
        if not os.path.isfile(script):
            raise FileNotFoundError(f"run_zorro.py not found at {script}")
        text = open(script, encoding='utf-8', errors='ignore').read()
        needed = {'--variant', '--attacks', '--model', '--cut'}
        for flags in self.variants.values():
            needed.update(f for f in flags if f.startswith('--'))
        missing = sorted(f for f in needed if f"'{f}'" not in text and f'"{f}"' not in text)
        if missing:
            raise RuntimeError(f"your run_zorro.py does not support {missing}; the ZORRO ablation needs them")
        return "'--dataset'" in text or '"--dataset"' in text

    def _launch(self, variant, attacks):
        script = os.path.join(C.ROOT, 'defence_runner', 'run_zorro.py')
        has_dataset = self._check_runner(script)
        cmd = [sys.executable, script, '--model', Config.MODEL_NAME, '--cut', str(Config.CUT_LAYER),
               '--variant', variant] + self.variants[variant]
        if has_dataset:
            cmd += ['--dataset', Config.DATASET]
        if attacks is not None:
            cmd += ['--attacks', ','.join(attacks)]
        print('[ablation] ' + ' '.join(cmd))
        subprocess.run(cmd, cwd=C.ROOT, check=True)

    def run(self, variant, attack, ctx):
        path = self._csv(variant)
        if variant not in self.done and not (ctx.get('reuse') and os.path.exists(path)):
            attacks = None if variant == 'full' else [ATTACK_NAME[a] for a in self.attacks]
            self._launch(variant, attacks)
            self.done.add(variant)
        df = pd.read_csv(path)
        row = df[df['attack'] == ATTACK_NAME[attack]]
        if row.empty or row.iloc[0]['status'] != 'ok':
            return {'error': 'attack failed in run_zorro.py'}
        r = row.iloc[0]
        out = {'asr': float(r['asr']), 'acc': float(r['cda'])}
        for k in ('zorro_PRR', 'zorro_BBR', 'zorro_proof_rejected', 'zorro_skipped_banned'):
            if k in r and pd.notna(r[k]):
                out[k.replace('zorro_', '')] = float(r[k])
        return out
