import os
import sys
import argparse
import traceback


def _find_dir(marker_file, subdirs):
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        for sub in subdirs:
            cand = os.path.join(d, sub)
            if os.path.isfile(os.path.join(cand, marker_file)):
                return cand
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(f"could not find {marker_file}")
        d = parent


PROJECT_ROOT = _find_dir('config.py', [''])
ABLATION_DIR = _find_dir('abl_common.py', ['ablation', os.path.join('defence_runner', 'ablation')])
RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (PROJECT_ROOT, RUNNER_DIR, ABLATION_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import torch

import abl_common as C
from config import Config
from abl_report import build_pdf
from abl_triguard import TriGuardEvaluation
from run_ablation import ablate, print_summary, merge_rows
from all_defences.triguard_defense import COMPONENT_SOURCES

MODELS = ['KAGN', 'PyramidCNN']
DATASET = 'CIFAR10'
DEFENSE_NAME = 'TriGuard-SL'


def intro_text(args):
    lines = [
        f"<b>{DEFENSE_NAME}</b> combines three components that the component-wise ablation found responsible for "
        "stopping specific attacks, chosen so that they do not conflict with each other.",
        f"<b>1. Secret sharing (from SplitFSS / Make Split, not Hijack).</b> {COMPONENT_SOURCES['secret_sharing']}. "
        "Each server only observes a uniformly random share of the smashed data. Targets: white-box inversion, "
        "AE decoder, FSHA.",
        f"<b>2. Backward Laplace noise (from R3eLU / Secure Split Learning).</b> {COMPONENT_SOURCES['backward_laplace']}. "
        f"Budget epsilon = {args.epsilon:g} ({args.sensitivity} sensitivity), gradient clipping bound C = {args.clip_c:g}. "
        "Randomized response, top-K selection, "
        "the dynamic budget and the forward noise of R3eLU are not used because the ablation found them not "
        "responsible or redundant with secret sharing. Target: label leakage.",
        f"<b>3. Private classifier heads (from SplitML).</b> {COMPONENT_SOURCES['private_heads']}. Each of the "
        f"{args.clients} clients keeps its own final classifier layer; the client-side layers are averaged with "
        "FedAvg and the server backbone is shared. Target: client-side backdoor poisoning.",
        "<b>Not included.</b> CKKS encryption (SplitML) is redundant with secret sharing; R3eLU forward noise "
        "destroyed accuracy and is redundant with secret sharing; ProtoGuard-SL prototype replacement and test-time "
        "filtering need the server to see embeddings and labels at test time, which conflicts with secret sharing "
        "and cannot be deployed.",
        "<b>Simulation note.</b> Secret sharing is simulated as in the SplitFSS ablation: training runs on the "
        "reconstructed values (functionality-preserving protocol), and each attack observes one server's share. "
        "Running the KAGN/PyramidCNN server layers inside a real MPC protocol is not implemented here.",
        "<b>Off.</b> In this report, <i>off</i> is the same federated split pipeline with all three components "
        "disabled (plain smashed data, no gradient noise, classifier heads averaged), so every component row "
        "isolates one component of TriGuard-SL.",
    ]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='both', help="KAGN, PyramidCNN or both")
    ap.add_argument('--cut', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--attacks', default='', help="comma list to restrict attacks")
    ap.add_argument('--variants', default='', help="comma list to restrict variants (full/off always run)")
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--fresh', action='store_true')
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--clients', type=int, default=3)
    ap.add_argument('--epsilon', type=float, default=1.0)
    ap.add_argument('--sensitivity', default='global', choices=['global', 'per_coord'])
    ap.add_argument('--clip-c', type=float, default=10.0, help="clipping bound for the cut-layer gradients")
    ap.add_argument('--out-subdir', default='', help="write results to results/triguard/<out-subdir>")
    ap.add_argument('--ae-epochs', type=int, default=C.SETTINGS['ae_epochs'])
    ap.add_argument('--fsha-epochs', type=int, default=C.SETTINGS['fsha_epochs'])
    ap.add_argument('--vanilla-epochs', type=int, default=C.SETTINGS['vanilla_epochs'])
    args = ap.parse_args()

    os.chdir(C.ROOT)
    args.attacks = [a for a in args.attacks.split(',') if a]
    args.variants = [v for v in args.variants.split(',') if v]
    models = MODELS if args.model == 'both' else [args.model]
    C.SETTINGS.update(ae_epochs=args.ae_epochs, fsha_epochs=args.fsha_epochs, vanilla_epochs=args.vanilla_epochs)
    if args.quick:
        C.SETTINGS.update(ae_epochs=2, fsha_epochs=1, wb_iters=20, max_images=4, leak_epochs=1,
                          vanilla_epochs=1, max_batches=5)
        args.rounds = 1

    suffix = '_quick' if args.quick else ''
    out_dir = os.path.join(Config.RESULTS_DIR, 'triguard', args.out_subdir) if args.out_subdir else os.path.join(Config.RESULTS_DIR, 'triguard')
    os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.join(out_dir, f"triguard_summary{suffix}.csv")
    pdf_path = os.path.join(out_dir, f"triguard_pdf{suffix}.pdf")
    meta = {'cut': args.cut, 'seed': args.seed, 'gates': C.GATES, 'headline': True,
            'title': f"{DEFENSE_NAME}: combined defense evaluation and component ablation",
            'intro': intro_text(args),
            'budgets': ', '.join(f"{k}={v}" for k, v in C.SETTINGS.items()) +
                       f", rounds={args.rounds}, clients={args.clients}, epsilon={args.epsilon:g}, "
                       f"sensitivity={args.sensitivity}, clip_c={args.clip_c:g}"}

    previous = [] if args.fresh or not os.path.exists(summary_csv) else pd.read_csv(summary_csv).to_dict('records')
    if args.report_only:
        print(f"[triguard] PDF -> {build_pdf(previous, pdf_path, meta)}")
        print_summary(previous)
        return

    all_rows = []
    combined = previous
    for model in models:
        device = C.setup(model, DATASET, args.cut, args.seed)
        ctx = {'device': device, 'seed': args.seed, 'cache': {}}
        ctx['train'], ctx['test'] = C.loaders()
        evaluation = TriGuardEvaluation(rounds=args.rounds, num_clients=args.clients, clip_c=args.clip_c,
                                        epsilon=args.epsilon, sensitivity=args.sensitivity)
        rows = []
        path = os.path.join(out_dir, f"triguard_{model.lower()}_{DATASET}_cut{args.cut}{suffix}.csv")
        try:
            ablate(evaluation, args, ctx, rows, path)
        except Exception:
            traceback.print_exc()
        all_rows += rows
        combined = merge_rows(previous, all_rows)
        pd.DataFrame(combined).to_csv(summary_csv, index=False)
        print(f"\n[triguard] saved -> {path}")
        del ctx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            print(f"[triguard] PDF -> {build_pdf(combined, pdf_path, meta)}")
        except Exception:
            traceback.print_exc()
    print_summary(all_rows)
    print(f"\n[triguard] final PDF: {pdf_path}")


if __name__ == '__main__':
    main()