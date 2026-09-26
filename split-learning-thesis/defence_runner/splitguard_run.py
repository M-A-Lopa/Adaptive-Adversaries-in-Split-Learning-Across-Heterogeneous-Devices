import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import random
import traceback
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import Config
from dataset import DatasetLoader
from all_model.models import ClientModel, ServerModel
from all_model.kagn_models import KAGNClientModel, KAGNServerModel
from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel
from all_split_learning.splitguard_split_learning import SplitGuardTrainer
from all_defences.splitguard_defense import (SplitGuardDefense, PAPER_POLICIES, SG_THRESHOLD,
                                             AF_TOLERANCE, A_ESTIMATORS)


MULT     = 5
EXP      = 2
B_FAKE   = 64
P_FAKE   = 0.1
N_START  = 20
EPOCHS   = 1
SEED     = 0

ADV_TYPES     = ['honest', 'random']
A_ESTIMATOR_S = ['output', 'local', 'linear']

STOP_POLICY = None
INCREASE_N  = True

DETECTION_POLICY = 'avg-10'

ATTACKS_TO_RUN = ["label_leakage", "villain", "poison_client", "poison_server",
                  "whitebox", "unsplit", "ae_decoder", "fsha"]

FSHA_CRITIC_ITERS = 5
RECON_IMAGES      = 32

POISON_TARGET  = 0
POISON_RATE    = 0.05
POISON_PATCH   = 4
POISON_TRIGGER = 1.0

QUICK_TEST = False

ATTACKS = {
    "label_leakage": ("Label Leakage (norm + cosine gradient scoring)", "client"),
    "villain":       ("VILLAIN backdoor (label inference + embedding trigger)", "client"),
    "poison_client": ("Backdoor poisoning, client attacker (BadNets patch)", "client"),
    "poison_server": ("Backdoor poisoning, server attacker (pre-poisoned server)", "server"),
    "whitebox":      ("White-box model inversion", "server-passive"),
    "unsplit":       ("UnSplit model inversion", "server-passive"),
    "ae_decoder":    ("AE decoder inversion", "server-passive"),
    "fsha":          ("FSHA feature-space hijacking", "server-hijack"),
}

CLEAN_CHECKPOINT_TAGS = {"Vanilla": "vanilla_sl", "PyramidCNN": "pyramidcnn_sl", "KAGN": "kagn_sl"}

DEFENSE_STAGE = "training: client sends fake-label batches and scores the server's gradients (SG score)"


SUCCESS, PARTIAL, FAILED, NOT_APPLICABLE = "SUCCESS", "PARTIAL", "FAILED", "N/A"
UTILITY_TOLERANCE_PCT = 5.0

SUMMARY_COLUMNS = [
    'dataset', 'model', 'cut_layer', 'attack', 'attack_name', 'defense', 'a_estimator',
    'attack_epochs', 'defense_epochs', 'defense_stage', 'attack_iterations', 'iteration_unit', 'defense_calls',
    'accuracy_no_defense', 'accuracy_with_defense', 'delta_accuracy',
    'metric', 'metric_no_defense', 'metric_with_defense', 'honest_flagged_at',
    'verdict', 'defense_working', 'reason',
]


def _missing(v):
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _fmt(v, spec, width):
    if _missing(v):
        return f"{'-':>{width}}"
    if isinstance(v, str):
        return f"{v:>{width}}"
    return f"{format(v, spec):>{width}}"


def detection_verdict(attack_key, detected_at, honest_at):
    policy = DETECTION_POLICY
    if ATTACKS[attack_key][1] == "server-passive":
        note = (f" (the honest-server run was also flagged at batch {honest_at}: a false alarm, not a detection)"
                if honest_at is not None else "")
        return FAILED, ("server trains honestly and only observes the smashed data, so its gradients are the same "
                        f"as an honest server's and SplitGuard cannot detect it{note}")
    if detected_at is None:
        return FAILED, f"not detected by the '{policy}' policy"
    if honest_at is not None:
        return PARTIAL, (f"flagged at batch {detected_at}, but the honest server is also flagged at batch {honest_at} "
                         f"(false positive), so '{policy}' cannot tell them apart")
    return SUCCESS, f"detected at batch {detected_at} by '{policy}'; honest server not flagged"


def finalize_row(row):
    a0, a1 = row.get('accuracy_no_defense'), row.get('accuracy_with_defense')
    row['delta_accuracy'] = float(a1) - float(a0) if not (_missing(a0) or _missing(a1)) else float('nan')
    row.setdefault('metric', 'Detect@')
    if row.get('verdict') == NOT_APPLICABLE:
        row['defense_working'] = "N/A"
        return row
    verdict, reason = detection_verdict(row['attack'], row.get('detected_at'), row.get('honest_flagged_at'))
    utility_ok = True
    if not _missing(row['delta_accuracy']):
        utility_ok = row['delta_accuracy'] >= -UTILITY_TOLERANCE_PCT
        reason += f"; accuracy {row['delta_accuracy']:+.2f} pts"
    if not _missing(row.get('ssim_no_defense')) and not _missing(row.get('ssim_with_defense')):
        reason += (f"; reconstruction SSIM {row['ssim_no_defense']:.3f} if training continues vs "
                   f"{row['ssim_with_defense']:.3f} if stopped at detection")
    row['verdict'] = verdict
    row['defense_working'] = {SUCCESS: "YES" if utility_ok else "YES (high acc. cost)",
                              PARTIAL: "PARTIAL", FAILED: "NO"}.get(verdict, verdict)
    row['reason'] = row.get('extra_reason', '') + reason if row.get('extra_reason') else reason
    return row


def print_summary_table(rows, title):
    line = "=" * 176
    print("\n" + line)
    print(f"   SUMMARY -- {title}")
    print(line)
    print(f"  {'Attack':<14} {'Defense':<18} {'Cut':>5} {'AtkEp':>6} {'DefEp':>6} {'Iterations':>11} "
          f"{'FakeBat':>8} {'Acc0%':>7} {'Acc%':>7} {'dAcc':>7} {'Detect@':>8} {'Honest@':>8} "
          f"{'SSIM0':>7} {'SSIM':>7}  {'Verdict':<9} {'Working':<20}")
    print("-" * 176)
    for r in rows:
        print(f"  {r['attack']:<14} {r['defense']:<18.18} {str(r['cut_layer']):>5} "
              f"{_fmt(r.get('attack_epochs'), '.0f', 6)} {_fmt(r.get('defense_epochs'), '.0f', 6)} "
              f"{_fmt(r.get('attack_iterations'), ',.0f', 11)} {_fmt(r.get('defense_calls'), ',.0f', 8)} "
              f"{_fmt(r.get('accuracy_no_defense'), '.2f', 7)} {_fmt(r.get('accuracy_with_defense'), '.2f', 7)} "
              f"{_fmt(r.get('delta_accuracy'), '+.2f', 7)} "
              f"{_fmt('none' if r.get('verdict') != NOT_APPLICABLE and r.get('detected_at') is None else r.get('detected_at'), '.0f', 8)} "
              f"{_fmt(r.get('honest_flagged_at'), '.0f', 8)} "
              f"{_fmt(r.get('ssim_no_defense'), '.3f', 7)} {_fmt(r.get('ssim_with_defense'), '.3f', 7)}  "
              f"{r['verdict']:<9} {r['defense_working']:<20}")
    print("-" * 176)
    print("  AtkEp = attack training epochs | DefEp = epochs SplitGuard was monitoring | Iterations = training steps "
          "| FakeBat = fake-label batches SplitGuard sent")
    print("  Acc0/Acc = test accuracy of honest training without/with SplitGuard | Detect@ = batch index where the "
          f"'{DETECTION_POLICY}' policy reported an attack | Honest@ = same policy on the honest server (false positive)")
    print("  SSIM0 = FSHA reconstruction if training continues | SSIM = FSHA reconstruction if training is stopped at "
          "detection")
    print("  SUCCESS = attack detected and honest server not flagged | PARTIAL = detected but honest also flagged | "
          "FAILED = not detected")
    print(line)


def print_defense_attack_matrix(rows):
    rows = [r for r in rows]
    attacks = list(dict.fromkeys(r['attack'] for r in rows))
    defenses = list(dict.fromkeys(r['defense'] for r in rows))
    cell = {(r['defense'], r['attack']): r['defense_working'] for r in rows}
    width = max(15, max(len(a) for a in attacks) + 2)
    print("\n" + "=" * (22 + width * len(attacks)))
    print("   WHICH SPLITGUARD SETTING WORKS AGAINST WHICH ATTACK")
    print("=" * (22 + width * len(attacks)))
    print(f"  {'Defense':<18}" + "".join(f"{a:>{width}}" for a in attacks))
    print("-" * (22 + width * len(attacks)))
    for d in defenses:
        print(f"  {d:<18.18}" + "".join(f"{cell.get((d, a), '.')[:width - 2]:>{width}}" for a in attacks))
    print("=" * (22 + width * len(attacks)))
    for d in defenses:
        mine = [r for r in rows if r['defense'] == d]
        works = [r['attack'] for r in mine if r['verdict'] == SUCCESS]
        partial = [r['attack'] for r in mine if r['verdict'] == PARTIAL]
        fails = [r['attack'] for r in mine if r['verdict'] == FAILED]
        print(f"  {d:<18} successful against: {', '.join(works) or '-'} | partial: {', '.join(partial) or '-'} | "
              f"fails: {', '.join(fails) or '-'}")


def save_results(rows, path):
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
        df['cut_layer'] = df['cut_layer'].astype(str)
        df = df.drop_duplicates(subset=['dataset', 'model', 'cut_layer', 'attack', 'defense'], keep='last')
    df = df[[c for c in SUMMARY_COLUMNS if c in df.columns] + [c for c in df.columns if c not in SUMMARY_COLUMNS]]
    df.to_csv(path, index=False)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def model_from_config():
    name = Config.MODEL_NAME.lower()
    if "kagn" in name:
        return "KAGN"
    if "pyramid" in name:
        return "PyramidCNN"
    return "Vanilla"


def build_split_models(model_name, in_channels, cut_layer, num_classes):
    if model_name == "KAGN":
        return (KAGNClientModel(cut_layer=cut_layer, in_channels=in_channels, degree=Config.DEGREE),
                KAGNServerModel(cut_layer=cut_layer, num_classes=num_classes, in_channels=in_channels,
                                degree=Config.DEGREE))
    if model_name == "PyramidCNN":
        return (PyramidCNNClientModel(cut_layer=cut_layer, in_channels=in_channels),
                PyramidCNNServerModel(cut_layer=cut_layer, num_classes=num_classes, in_channels=in_channels))
    return ClientModel(in_channels=in_channels), ServerModel(num_classes=num_classes)


def make_defense(p_fake=P_FAKE):
    return SplitGuardDefense(b_fake=B_FAKE, p_fake=p_fake, N=N_START, mult=MULT, exp=EXP,
                             num_classes=Config.NUM_CLASSES, af_tolerance=AF_TOLERANCE,
                             increase_n=INCREASE_N)


def materialise(client, server, in_channels):
    size = 28 if Config.DATASET == 'MNIST' else 32
    device = next(server.parameters()).device if any(True for _ in server.parameters()) else torch.device('cpu')
    client.to(device)
    with torch.no_grad():
        server(client(torch.zeros(2, in_channels, size, size, device=device)))


if __name__ == "__main__":
    if QUICK_TEST:
        FSHA_CRITIC_ITERS, RECON_IMAGES = 1, 4

    dataset = DatasetLoader(dataset_name=Config.DATASET)
    train_loader, test_loader = dataset.get_loaders()
    in_channels = 1 if Config.DATASET == 'MNIST' else 3

    model_name = model_from_config()
    cut_layer = Config.CUT_LAYER
    cut_label = "fixed" if model_name == "Vanilla" else cut_layer
    iterations = len(train_loader) * EPOCHS
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    print("=" * 78)
    print("  SPLITGUARD DEFENSE vs ALL ATTACKS")
    print(f"  Dataset   : {Config.DATASET}")
    print(f"  Model     : {model_name}  (Config.MODEL_NAME = {Config.MODEL_NAME})")
    print(f"  Cut layer : {cut_label}")
    print(f"  Estimators: {A_ESTIMATOR_S} | detection policy for verdicts: {DETECTION_POLICY}")
    print(f"  Attacks   : {ATTACKS_TO_RUN}")
    print("=" * 78)

    results, summary, trainers = {}, [], {}

    for a_est in A_ESTIMATOR_S:
        for adv_type in ADV_TYPES:
            random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

            client_model, server_model = build_split_models(model_name, in_channels, cut_layer, Config.NUM_CLASSES)
            materialise(client_model, server_model, in_channels)
            defense = make_defense()

            trainer = SplitGuardTrainer(client_model, server_model, train_loader, test_loader,
                                        defense=defense, adv_type=adv_type,
                                        stop_policy=STOP_POLICY, a_estimator=a_est)
            results[(a_est, adv_type)] = trainer.train(epochs=EPOCHS)
            trainer.save_results()
            trainers[(a_est, adv_type)] = trainer

            summary.append({'a_estimator': a_est,
                            'server': adv_type,
                            'fake_batches': len(defense.fakes),
                            'mean_sg_score': defense.mean_score(),
                            'final_sg_score': defense.scores[-1] if defense.scores else float('nan'),
                            'final_A': trainer.estimator.accuracy(len(train_loader) - 1),
                            'test_accuracy': trainer.test_accuracies[-1],
                            'stopped_at_batch': trainer.stopped_at,
                            'final_b_fake': defense.b_fake,
                            'final_N': defense.N,
                            **{f'detect_{p}': defense.detections[p] for p in PAPER_POLICIES},
                            'dataset': Config.DATASET,
                            'model': model_name,
                            'cut_layer': cut_label,
                            'epochs': EPOCHS,
                            'iterations': iterations})

    for adv_type in ADV_TYPES:
        key = (A_ESTIMATOR_S[0], adv_type)
        print(f'{adv_type} mean: {np.mean(results[key])}')
        plt.plot(results[key], label=f'{adv_type}')
    plt.ylim(0, 1.1)
    plt.xlabel('No. of fake batches')
    plt.ylabel('SG score')
    plt.title(f'SplitGuard -- {Config.DATASET}')
    plt.legend()
    plot_path = f"{Config.RESULTS_DIR}/splitguard_scores_{Config.DATASET}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    df = pd.DataFrame(summary)
    out = f"{Config.RESULTS_DIR}/splitguard_summary_{Config.DATASET}.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 85)
    print(f"   SPLITGUARD DETECTION -- {Config.DATASET}")
    print("=" * 85)
    print(f"{'A estimate':<11} {'Server':<8} {'Fake batches':>13} {'Mean SG':>9} {'Final SG':>9} {'Final A':>9} {'Test Acc (%)':>13}")
    print("-" * 85)
    for r in summary:
        print(f"{r['a_estimator']:<11} {r['server']:<8} {r['fake_batches']:>13} {r['mean_sg_score']:>9.4f} "
              f"{r['final_sg_score']:>9.4f} {r['final_A']:>9.4f} {r['test_accuracy']:>13.2f}")
    print("=" * 85)

    print("\n" + "=" * 85)
    print(f"   ALGORITHM 3 DECISIONS (SG threshold {SG_THRESHOLD}, A ~ A_F tolerance {AF_TOLERANCE}) -- detection batch index")
    print("   honest row = false positive if detected | random row = true positive if detected")
    print("=" * 85)
    print(f"{'A estimate':<11} {'Server':<8}" + "".join(f"{p:>12}" for p in PAPER_POLICIES))
    print("-" * 85)
    for r in summary:
        cells = "".join(f"{('-' if r[f'detect_{p}'] is None else r[f'detect_{p}']):>12}" for p in PAPER_POLICIES)
        print(f"{r['a_estimator']:<11} {r['server']:<8}{cells}")
    print("=" * 85)
    for r in summary:
        if r['final_N'] != N_START or r['final_b_fake'] != B_FAKE:
            print(f"  [{r['a_estimator']} | {r['server']}] Algorithm 3 adjusted B_F -> {r['final_b_fake']}/64, N -> {r['final_N']}")
    if STOP_POLICY:
        for r in summary:
            print(f"  [{r['a_estimator']} | {r['server']}] stopped by '{STOP_POLICY}' at batch: {r['stopped_at_batch']}")
    print(f"\nSaved -> {out}\nPlot  -> {plot_path}")

    print("\n" + "#" * 78)
    print("#   SPLITGUARD vs ALL ATTACKS")
    print(f"#   MODEL: {model_name} | CUT LAYER: {cut_label} | {Config.DATASET}")
    print("#" * 78)

    print("\n  >> Baseline: honest training without SplitGuard (for accuracy comparison)")
    set_seed()
    c0, s0 = build_split_models(model_name, in_channels, cut_layer, Config.NUM_CLASSES)
    materialise(c0, s0, in_channels)
    base_trainer = SplitGuardTrainer(c0, s0, train_loader, test_loader, defense=make_defense(p_fake=0.0),
                                     adv_type='honest', a_estimator='output')
    base_trainer.train(epochs=EPOCHS)
    baseline_acc = base_trainer.test_accuracies[-1]

    def base_row(attack_key, a_est):
        return {"dataset": Config.DATASET, "model": model_name, "cut_layer": cut_label, "attack": attack_key,
                "attack_name": ATTACKS[attack_key][0], "defense": f"splitguard_{a_est}", "a_estimator": a_est,
                "attack_epochs": EPOCHS, "defense_epochs": EPOCHS, "defense_stage": DEFENSE_STAGE,
                "attack_iterations": iterations, "iteration_unit": "training steps (train batches x epochs)",
                "accuracy_no_defense": baseline_acc, "metric": "Detect@", "metric_no_defense": "-"}

    def from_trainer(row, trainer):
        d = trainer.defense
        row.update({"defense_calls": len(d.fakes), "mean_sg_score": d.mean_score(),
                    "detected_at": d.detections[DETECTION_POLICY],
                    "metric_with_defense": d.detections[DETECTION_POLICY],
                    "final_b_fake": d.b_fake, "final_N": d.N,
                    **{f"detect_{p}": d.detections[p] for p in PAPER_POLICIES}})
        return row

    def fsha_metrics(fsha, images):
        from all_attacks.fsha_attack import compute_mse, compute_psnr, compute_ssim, denormalize
        was_c, was_d = fsha.client_model.training, fsha.decoder.training
        fsha.client_model.eval(); fsha.decoder.eval()
        with torch.no_grad():
            rec = fsha.decoder(fsha.client_model(images)).clamp(0, 1)
            orig = denormalize(images.cpu(), Config.DATASET).to(rec.device)
            ssim = [compute_ssim(orig[i].unsqueeze(0), rec[i].unsqueeze(0)) for i in range(len(images))]
            psnr = [compute_psnr(orig[i], rec[i]) for i in range(len(images))]
            mse = [compute_mse(orig[i], rec[i]) for i in range(len(images))]
        fsha.client_model.train(was_c); fsha.decoder.train(was_d)
        return float(np.mean(ssim)), float(np.mean(psnr)), float(np.mean(mse))

    eval_images = []
    for x, _ in train_loader:
        eval_images.append(x)
        if sum(len(e) for e in eval_images) >= RECON_IMAGES:
            break
    eval_images = torch.cat(eval_images)[:RECON_IMAGES]

    all_rows = []
    all_out = f"{Config.RESULTS_DIR}/splitguard_all_attacks_{Config.DATASET}.csv"

    for a_est in A_ESTIMATOR_S:
        if (a_est, 'honest') in trainers:
            honest = trainers[(a_est, 'honest')]
        else:
            set_seed()
            ch, sh = build_split_models(model_name, in_channels, cut_layer, Config.NUM_CLASSES)
            materialise(ch, sh, in_channels)
            honest = SplitGuardTrainer(ch, sh, train_loader, test_loader, defense=make_defense(),
                                       adv_type='honest', a_estimator=a_est)
            honest.train(epochs=EPOCHS)
        honest_at = honest.defense.detections[DETECTION_POLICY]

        for attack_key in ATTACKS_TO_RUN:
            print("\n" + "#" * 78)
            print(f"#   SPLITGUARD ({a_est}) vs {attack_key} -- {ATTACKS[attack_key][0]}")
            print(f"#   MODEL: {model_name} | CUT LAYER: {cut_label} | {Config.DATASET}")
            print("#" * 78)
            row = base_row(attack_key, a_est)
            row["honest_flagged_at"] = honest_at
            try:
                side = ATTACKS[attack_key][1]
                if side == "client":
                    row.update({"verdict": NOT_APPLICABLE, "defense_epochs": 0, "defense_calls": 0,
                                "attack_iterations": float('nan'), "attack_epochs": float('nan'),
                                "accuracy_no_defense": float('nan'), "metric_no_defense": float('nan'),
                                "reason": "the client is the attacker; SplitGuard protects the client against a "
                                          "malicious server"})
                    print(f"  [N/A] {row['reason']}")

                elif side == "server-passive":
                    from_trainer(row, honest)
                    row["accuracy_with_defense"] = honest.test_accuracies[-1]
                    row["detected_at"] = None
                    row["metric_with_defense"] = "none"
                    print("  Server trains honestly (only observes smashed data): using the honest-server "
                          "SplitGuard run.")

                elif attack_key == "poison_server":
                    from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack
                    set_seed()
                    cp, sp = build_split_models(model_name, in_channels, cut_layer, Config.NUM_CLASSES)
                    atk = BackdoorPoisonAttack(client_model=cp, server_model=sp,
                                               base_dataset=train_loader.dataset, dataset=Config.DATASET,
                                               num_classes=Config.NUM_CLASSES, mode="server",
                                               target_label=POISON_TARGET, poison_rate=POISON_RATE,
                                               patch_size=POISON_PATCH, trigger_value=POISON_TRIGGER,
                                               surrogate_builder=lambda: build_split_models(
                                                   model_name, in_channels, cut_layer, Config.NUM_CLASSES)[0],
                                               model_tag=CLEAN_CHECKPOINT_TAGS[model_name])
                    print(f"  Poisoned checkpoint path: {atk._checkpoint_path()}")
                    if not atk.load_checkpoint():
                        print(f"\n[!] No poisoned checkpoint found at: {atk._checkpoint_path()}")
                        print("    Run defence_runner/run_backdoor_poison.py first to produce it. Skipping.")
                        row.update({"verdict": NOT_APPLICABLE, "defense_calls": 0, "defense_epochs": 0,
                                    "attack_iterations": float('nan'), "attack_epochs": float('nan'),
                                    "accuracy_no_defense": float('nan'), "metric_no_defense": float('nan'),
                                    "reason": f"no poisoned checkpoint at {atk._checkpoint_path()}"})
                    else:
                        fresh_client = build_split_models(model_name, in_channels, cut_layer, Config.NUM_CLASSES)[0]
                        materialise(fresh_client, atk.server_model, in_channels)
                        trainer = SplitGuardTrainer(fresh_client, atk.server_model, train_loader, test_loader,
                                                    defense=make_defense(), adv_type='honest',
                                                    stop_policy=STOP_POLICY, a_estimator=a_est)
                        trainer.train(epochs=EPOCHS)
                        atk.client_model = trainer.client_model
                        cda, asr = atk.evaluate(test_loader)
                        from_trainer(row, trainer)
                        row.update({"accuracy_with_defense": cda, "poison_asr_after_training": asr,
                                    "extra_reason": f"pre-poisoned server trained honestly under SplitGuard "
                                                    f"(ASR afterwards {asr:.1f}%); "})

                else:
                    from all_attacks.fsha_attack import FSHAAttack
                    set_seed()
                    cf, sf = build_split_models(model_name, in_channels, cut_layer, Config.NUM_CLASSES)
                    materialise(cf, sf, in_channels)
                    fsha = FSHAAttack(client_model=cf, in_channels=in_channels, dataset=Config.DATASET,
                                      pilot_builder=lambda: build_split_models(
                                          model_name, in_channels, cut_layer, Config.NUM_CLASSES)[0],
                                      critic_iters=FSHA_CRITIC_ITERS)
                    images = eval_images.to(fsha.device)
                    at_detection = {}

                    def on_detection(policy, index):
                        if policy == DETECTION_POLICY and not at_detection:
                            at_detection['metrics'] = fsha_metrics(fsha, images)
                            print(f"\n  [SplitGuard] '{policy}' flagged the FSHA server at batch {index}")

                    trainer = SplitGuardTrainer(cf, sf, train_loader, test_loader, defense=make_defense(),
                                                adv_type='fsha', stop_policy=STOP_POLICY, a_estimator=a_est,
                                                server_attack=fsha, public_loader=test_loader,
                                                on_detection=on_detection)
                    results[(a_est, 'fsha')] = trainer.train(epochs=EPOCHS)
                    trainer.save_results()
                    final = fsha_metrics(fsha, images)
                    stopped = at_detection.get('metrics', final)
                    from_trainer(row, trainer)
                    row.update({"accuracy_no_defense": float('nan'), "accuracy_with_defense": float('nan'),
                                "ssim_no_defense": final[0], "psnr_no_defense": final[1], "mse_no_defense": final[2],
                                "ssim_with_defense": stopped[0], "psnr_with_defense": stopped[1],
                                "mse_with_defense": stopped[2]})
            except Exception as exc:
                traceback.print_exc()
                row.update({"verdict": NOT_APPLICABLE, "reason": f"run failed: {type(exc).__name__}: {exc}"})

            finalize_row(row)
            print(f"\n  [run info] attack={attack_key} | cut={cut_label} | attack epochs={row.get('attack_epochs')} | "
                  f"defense epochs={row.get('defense_epochs')} | iterations={row.get('attack_iterations')} | "
                  f"fake batches={row.get('defense_calls')}")
            print(f"  Result: {row['defense_working']} -- {row.get('reason', '')}")
            all_rows.append(row)
            save_results(all_rows, all_out)

    print_summary_table(all_rows, f"SPLITGUARD vs ALL ATTACKS -- {model_name} | cut {cut_label} | {Config.DATASET}")
    print_defense_attack_matrix(all_rows)
    save_results(all_rows, all_out)
    print(f"\n  Saved -> {all_out}")
