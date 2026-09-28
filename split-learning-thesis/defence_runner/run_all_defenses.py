import os
import re
import sys
import json
import time
import math
import shutil
import argparse
import subprocess
from datetime import datetime
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


DATASET          = "CIFAR10"
MODELS           = ["PyramidCNN", "KAGN"]
CUT_LAYERS       = [4]
DEFENSES_TO_RUN  = ["label_protection", "protoguard", "splitguard", "r3elu", "splitfss", "splitml"]

QUICK_TEST           = False
RESUME               = True
PREPARE_CHECKPOINTS  = True
APPLY_RUNNER_FIXES   = True

VANILLA_EPOCHS            = None
BACKDOOR_TARGET_LABEL     = 0
BACKDOOR_POISON_RATE      = 0.05
BACKDOOR_PATCH_SIZE       = 4
BACKDOOR_TRIGGER_VALUE    = 1.0
BACKDOOR_TRAIN_EPOCHS     = 10
BACKDOOR_SURROGATE_EPOCHS = 5

RUNNER_OVERRIDES = {}


DEFENSES = {
    "label_protection": ("Label Protection", "run_label_protection.py",     "label_protection_all_attacks",
                         ["marvell_s1.75"]),
    "protoguard":       ("ProtoGuard",       "protoguard_run.py",           "protoguard_sl_all_attacks",
                         ["protoguard_a0.5"]),
    "splitguard":       ("SplitGuard",       "splitguard_run.py",           "splitguard_all_attacks",
                         ["splitguard_output"]),
    "r3elu":            ("R3eLU",            "run_r3elu_evaluation.py",     "r3elu_all_attacks",
                         ["r3elu_eps1.0"]),
    "splitfss":         ("SplitFSS",         "run_splitfss_evaluation.py",  "splitfss_all_attacks",
                         ["splitfss_P0"]),
    "splitml":          ("SplitML",          "run_splitml_evaluation.py",   "splitml_all_attacks",
                         ["splitml_peer", "splitml_federation"]),
}

RUNNER_FIXES = {"run_r3elu_evaluation.py": "fsha", "run_splitfss_evaluation.py": "fsha"}
FSHA_FIX = [
    ("recon_loss = self.mse(self.decoder(self.pilot(pub)), pub)",
     "recon_loss = self.mse(self.decoder(self.pilot(pub)), fsha_denormalize(pub, self.dataset))"),
    ("n = min(real.shape[0], fake.shape[0])", "nb = min(real.shape[0], fake.shape[0])"),
    ("gp = gradient_penalty(self.critic, real[:n], fake[:n], self.device)",
     "gp = gradient_penalty(self.critic, real[:nb], fake[:nb], self.device)"),
    ("critic_loss = self.critic(fake[:n]).mean() - self.critic(real[:n]).mean() + self.gp_lambda * gp",
     "critic_loss = self.critic(fake[:nb]).mean() - self.critic(real[:nb]).mean() + self.gp_lambda * gp"),
]

ATTACK_ORDER = ["whitebox", "unsplit", "ae_decoder", "fsha", "label_leakage", "villain",
                "poison_client", "poison_server"]
ATTACK_LABEL = {"whitebox": "White-Box", "unsplit": "UnSplit", "ae_decoder": "AE Decoder", "fsha": "FSHA",
                "label_leakage": "Label Leakage", "villain": "VILLAIN",
                "poison_client": "Backdoor(Client)", "poison_server": "Backdoor(Server)"}
RECON = {"whitebox", "unsplit", "ae_decoder", "fsha"}
BACKDOOR = {"villain", "poison_client", "poison_server"}
LOG_MARKERS = ["THESIS TABLE", "SUMMARY --", "SPLITGUARD DETECTION", "PHASE A", "ALGORITHM 3"]
PROGRESS_LINES_PER_RUN = 4000
EPOCH_RE = re.compile(r"\b(epoch|round)\s*\[?\s*\d+\s*/\s*\d+", re.I)
ITER_RE = re.compile(r"Iteration (\d+)/(\d+)")
CONTEXT_RE = re.compile(r"^\s*(#   |>> |Phase \d|Running |\[run info\]|Result:|Training complete|Collected |"
                        r"Smashed data shape|Clean data accuracy|Hijacking complete|\[SplitGuard\]|"
                        r"Accuracy WITH defense|BASELINE|R3eLU DEFENSE|SplitFSS|CONSENSUS|Poisoned checkpoint)")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

DASH = "\u2014"
VERDICT_COLORS = {"SUCCESS": "#d4edda", "PARTIAL": "#fff3cd", "FAILED": "#f4c7c3", "ATTACK-WEAK": "#e2e3e5"}
CUT = None


def read_csv(path):
    import pandas as pd
    return pd.read_csv(path, keep_default_na=False, na_values=["", "nan", "NaN"])


def num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def text_or_dash(v):
    if num(v) is not None:
        return str(v)
    s = "" if v is None else str(v).strip()
    return DASH if s.lower() in ("", "nan", "-") else s


def fmt(v, spec, na=DASH):
    f = num(v)
    return na if f is None else format(f, spec)


def clean_text(v):
    s = "" if v is None else str(v)
    return "" if s in ("None", "nan") else s


def project_path(d):
    return os.path.normpath(d if os.path.isabs(d) else os.path.join(ROOT, d))


def results_dir():
    from config import Config
    return project_path(Config.RESULTS_DIR)


def out_dir():
    d = os.path.join(results_dir(), "all_defenses")
    os.makedirs(os.path.join(d, "raw"), exist_ok=True)
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    return d


def snapshot_path(key, model, cut):
    return os.path.join(out_dir(), "raw", f"{key}_{model.lower()}_cut{cut}_{DATASET}.csv")


def log_path(key, model, cut):
    return os.path.join(out_dir(), "logs", f"{key}_{model.lower()}_cut{cut}_{DATASET}.log")


def status_path():
    return os.path.join(out_dir(), f"run_status_{DATASET}.json")


def load_status():
    if os.path.exists(status_path()):
        with open(status_path(), encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_status(status):
    with open(status_path(), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


def status_key(model, cut, key):
    return f"{model}|{cut}|{key}"


BOOTSTRAP = r"""
import os, sys, json, re
job = json.loads(os.environ["ALL_DEFENSES_JOB"])
sys.path.insert(0, job["root"])
from config import Config
Config.DATASET = job["dataset"]
Config.MODEL_NAME = job["model"]
Config.CUT_LAYER = job["cut"]
if job["mode"] == "prepare":
    sys.path.insert(0, job["here"])
    import importlib.util
    spec = importlib.util.spec_from_file_location("orchestrator", job["self"])
    orchestrator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orchestrator)
    orchestrator.DATASET = job["dataset"]
    orchestrator.prepare_checkpoints()
    sys.exit(0)
path = job["runner"]
src = open(path, encoding="utf-8").read()
for name, value in job["overrides"].items():
    src, n = re.subn(rf"^{re.escape(name)}\s*=.*$", f"{name} = {value}", src, count=1, flags=re.M)
    print(f"[run_all_defenses] override {name} = {value}" + ("" if n else "  (NOT FOUND in runner)"))
for old, new in job["fixes"]:
    if src.count(old) == 1:
        src = src.replace(old, new)
        print(f"[run_all_defenses] runner fix applied: {new[:70]}")
    else:
        print(f"[run_all_defenses] runner fix skipped (pattern not found once): {old[:70]}")
sys.argv = [path]
sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path, "__builtins__": __builtins__})
"""


def run_child(job, log_file):
    env = dict(os.environ, ALL_DEFENSES_JOB=json.dumps(job), PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
               PYTHONIOENCODING="utf-8")
    start = time.time()
    with open(log_file, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-c", BOOTSTRAP], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    return proc.returncode, time.time() - start


def base_job(model, cut, mode):
    return {"root": ROOT, "here": HERE, "self": os.path.abspath(__file__), "dataset": DATASET, "model": model,
            "cut": cut, "mode": mode, "runner": None, "overrides": {}, "fixes": []}


def prepare_checkpoints():
    import torch
    from config import Config
    from dataset import DatasetLoader
    from all_model.models import ClientModel, ServerModel
    from all_model.kagn_models import KAGNClientModel, KAGNServerModel
    from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    in_ch = 1 if Config.DATASET == "MNIST" else 3
    size = 28 if Config.DATASET == "MNIST" else 32
    model = Config.MODEL_NAME
    tag = {"KAGN": "kagn_sl", "PyramidCNN": "pyramidcnn_sl"}.get(model, "vanilla_sl")
    cut_aware = model in ("KAGN", "PyramidCNN")
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    def build(num_classes=Config.NUM_CLASSES):
        if model == "KAGN":
            return (KAGNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_ch, degree=Config.DEGREE),
                    KAGNServerModel(cut_layer=Config.CUT_LAYER, num_classes=num_classes, in_channels=in_ch,
                                    degree=Config.DEGREE))
        if model == "PyramidCNN":
            return (PyramidCNNClientModel(cut_layer=Config.CUT_LAYER, in_channels=in_ch),
                    PyramidCNNServerModel(cut_layer=Config.CUT_LAYER, num_classes=num_classes, in_channels=in_ch))
        return ClientModel(in_channels=in_ch), ServerModel(num_classes=num_classes)

    def materialise(c, s):
        c.to(device)
        s.to(device)
        c.eval()
        s.eval()
        with torch.no_grad():
            s(c(torch.zeros(2, in_ch, size, size, device=device)))
        c.train()
        s.train()

    train_loader, test_loader = DatasetLoader(dataset_name=Config.DATASET).get_loaders()

    vanilla = f"{Config.SAVE_DIR}/best_{model.lower()}_sl_{Config.DATASET}.pth"
    per_cut = f"{Config.SAVE_DIR}/best_{model.lower()}_sl_cut{Config.CUT_LAYER}_{Config.DATASET}.pth"
    ready = False
    if cut_aware and os.path.exists(per_cut):
        shutil.copyfile(per_cut, vanilla)
        print(f"[prepare] clean checkpoint for cut {Config.CUT_LAYER} restored: {per_cut} -> {vanilla}")
        ready = True
    elif os.path.exists(vanilla):
        ck = torch.load(vanilla, map_location="cpu", weights_only=False)
        ready = (not cut_aware) or ck.get("cut_layer", Config.CUT_LAYER) == Config.CUT_LAYER
        if ready and cut_aware:
            shutil.copyfile(vanilla, per_cut)
        print(f"[prepare] clean checkpoint {vanilla} "
              + ("found." if ready else f"is for cut {ck.get('cut_layer')} -- training one for cut {Config.CUT_LAYER}"))
    if not ready:
        from all_split_learning.split_learning import SplitLearningTrainer
        if VANILLA_EPOCHS is not None:
            Config.EPOCHS = VANILLA_EPOCHS
        print(f"[prepare] training clean {model} SL for {Config.EPOCHS} epochs (cut {Config.CUT_LAYER})...")
        c, s = build()
        materialise(c, s)
        trainer = SplitLearningTrainer(client_model=c, server_model=s, train_loader=train_loader,
                                       test_loader=test_loader)
        trainer.train()
        trainer.save_results()
        produced = f"{Config.SAVE_DIR}/best_{Config.MODEL_NAME.lower()}_{Config.DATASET}.pth"
        ck = torch.load(produced, map_location="cpu", weights_only=False) if os.path.exists(produced) else None
        if ck is None:
            ck = {"client_state": c.state_dict(), "server_state": s.state_dict(),
                  "best_acc": max(trainer.test_accuracies), "epoch": Config.EPOCHS - 1}
        ck.update({"cut_layer": Config.CUT_LAYER, "dataset": Config.DATASET})
        torch.save(ck, vanilla)
        if cut_aware:
            torch.save(ck, per_cut)
        print(f"[prepare] saved clean checkpoint -> {vanilla} (best acc {ck.get('best_acc', float('nan')):.2f}%)")

    from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack
    for mode in ("client", "server"):
        c, s = build()
        materialise(c, s)
        surrogate = (lambda: build()[0]) if mode == "server" else None
        atk = BackdoorPoisonAttack(client_model=c, server_model=s, base_dataset=train_loader.dataset,
                                   dataset=Config.DATASET, num_classes=Config.NUM_CLASSES, mode=mode,
                                   target_label=BACKDOOR_TARGET_LABEL, poison_rate=BACKDOOR_POISON_RATE,
                                   patch_size=BACKDOOR_PATCH_SIZE, trigger_value=BACKDOOR_TRIGGER_VALUE,
                                   surrogate_builder=surrogate, model_tag=tag)
        if os.path.exists(atk._checkpoint_path()):
            print(f"[prepare] poisoned {mode} checkpoint found: {atk._checkpoint_path()}")
            continue
        print(f"[prepare] creating poisoned {mode} checkpoint: {atk._checkpoint_path()}")
        atk.load_clean_init(vanilla)
        if mode == "server":
            atk.pretrain_server_backdoor(test_loader, epochs=BACKDOOR_SURROGATE_EPOCHS)
        atk.train(train_loader, test_loader, epochs=BACKDOOR_TRAIN_EPOCHS)
        atk.summarise()


def runner_csv(key):
    return os.path.join(results_dir(), f"{DEFENSES[key][2]}_{DATASET}.csv")


def same_run(df, model, cut):
    import pandas as pd
    if df.empty or "model" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return (df["model"].astype(str) == model) & (df["cut_layer"].astype(str) == str(cut))


def drop_stale_rows(key, model, cut):
    path = runner_csv(key)
    if os.path.exists(path):
        df = read_csv(path)
        df[~same_run(df, model, cut)].to_csv(path, index=False)


def collect(key, model, cut, seconds=None, code=None):
    import pandas as pd
    path = runner_csv(key)
    df = read_csv(path) if os.path.exists(path) else pd.DataFrame()
    df = df[same_run(df, model, cut)].copy() if not df.empty else df
    df["defense_key"] = key
    df["defense_name"] = DEFENSES[key][0]
    df["runtime_min"] = None if seconds is None else seconds / 60
    df["exit_code"] = code
    df.to_csv(snapshot_path(key, model, cut), index=False)
    print(f"[run_all_defenses] collected {len(df)} row(s) -> {snapshot_path(key, model, cut)}")
    return len(df)


def run_defense(key, model, cut, status):
    name, runner, _, _ = DEFENSES[key]
    runner_path = os.path.join(HERE, runner)
    sk = status_key(model, cut, key)
    if not os.path.exists(runner_path):
        print(f"[!] runner not found: {runner_path} -- skipping {name}")
        status[sk] = {"status": "runner missing", "log": "", "finished": now()}
        return False
    drop_stale_rows(key, model, cut)
    job = base_job(model, cut, "run")
    job["runner"] = runner_path
    job["overrides"] = dict(RUNNER_OVERRIDES.get(key, {}))
    if QUICK_TEST:
        job["overrides"]["QUICK_TEST"] = "True"
    if APPLY_RUNNER_FIXES and RUNNER_FIXES.get(runner) == "fsha":
        job["fixes"] = FSHA_FIX
    banner(f"{name}  |  {model}  |  cut {cut}  |  {DATASET}")
    status[sk] = {"status": "running", "started": now(), "log": log_path(key, model, cut)}
    save_status(status)
    code, seconds = run_child(job, log_path(key, model, cut))
    n_rows = collect(key, model, cut, seconds, code)
    status[sk].update({"status": "done" if code == 0 else "failed", "exit_code": code,
                       "runtime_min": round(seconds / 60, 1), "rows": n_rows, "finished": now()})
    save_status(status)
    print(f"\n[run_all_defenses] {name} / {model} / cut {cut} finished with exit code {code} "
          f"in {seconds / 60:.1f} min")
    return code == 0


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def load_rows(key, model, cut):
    path = snapshot_path(key, model, cut)
    if not os.path.exists(path):
        return []
    df = read_csv(path)
    return df.astype(object).where(df.notna(), None).to_dict("records")


def pick_rows(key, rows):
    reps = DEFENSES[key][3]
    chosen = {}
    for attack in ATTACK_ORDER:
        mine = [r for r in rows if r.get("attack") == attack and r.get("verdict") != "BASELINE"]
        pref = [r for rep in reps for r in mine if str(r.get("defense")) == rep]
        if pref or mine:
            chosen[attack] = (pref or mine)[0]
    return chosen


def extract_lia(row):
    for k in ("lia", "lia_with_defense"):
        if num(row.get(k)) is not None:
            return num(row[k])
    for k in ("reason", "note"):
        m = re.search(r"label inference ([\d.]+)% -> ([\d.]+)%", str(row.get(k) or ""))
        if m:
            return float(m.group(2))
    return None


def cells_for(attack, row):
    c = dict.fromkeys(["psnr", "ssim", "mse", "leak", "lia", "asr", "cda"])
    if row is None:
        return c, "NOT RUN"
    verdict = str(row.get("verdict") or "")
    if verdict == "N/A":
        return c, verdict
    detect = str(row.get("metric")) == "Detect@"
    if attack in RECON:
        c["psnr"] = row.get("psnr_with_defense")
        c["ssim"] = row.get("ssim_with_defense") if detect else row.get("metric_with_defense")
        c["mse"] = row.get("mse_with_defense")
    elif attack == "label_leakage":
        c["leak"] = row.get("metric_with_defense")
    else:
        c["asr"] = row.get("poison_asr_after_training") if detect else row.get("metric_with_defense")
        c["cda"] = row.get("accuracy_with_defense")
        if attack == "villain":
            c["lia"] = extract_lia(row)
    return c, verdict


def attack_table(key, rows):
    chosen = pick_rows(key, rows)
    table = []
    for attack in ATTACK_ORDER:
        row = chosen.get(attack)
        c, verdict = cells_for(attack, row)
        table.append({"attack": attack, "label": ATTACK_LABEL[attack], "row": row, "verdict": verdict, **c})
    return table


def screenshot_cells(t):
    na = "N/A" if t["verdict"] in ("N/A", "NOT RUN") else DASH
    rec, bd = t["attack"] in RECON, t["attack"] in BACKDOOR
    return [fmt(t["psnr"], ".2f", na if rec else DASH), fmt(t["ssim"], ".4f", na if rec else DASH),
            fmt(t["mse"], ".5f"), fmt(t["leak"], ".4f", na if t["attack"] == "label_leakage" else DASH),
            fmt(t["lia"], ".2f", na if t["attack"] == "villain" else DASH),
            fmt(t["asr"], ".2f", na if bd else DASH), fmt(t["cda"], ".2f", na if bd else DASH)]


def leakage_detail(key, rows):
    row = pick_rows(key, rows).get("label_leakage")
    if row is None or row.get("verdict") == "N/A":
        return None
    return [("Norm (cut layer)", row.get("norm_leak_auc_cut")),
            ("Cosine (cut layer)", row.get("cosine_leak_auc_cut")),
            ("Norm (first layer)", row.get("norm_leak_auc_first")),
            ("Cosine (first layer)", row.get("cosine_leak_auc_first")),
            ("Majority counting accuracy", row.get("majority_counting_acc_q95"))]


def accuracy_row(key, rows, status_entry):
    chosen = pick_rows(key, rows)
    main0 = main1 = None
    for attack in ("whitebox", "unsplit", "ae_decoder", "fsha"):
        r = chosen.get(attack)
        if r is not None and num(r.get("accuracy_with_defense")) is not None:
            main0, main1 = num(r.get("accuracy_no_defense")), num(r.get("accuracy_with_defense"))
            break
    out = {"defense": DEFENSES[key][0],
           "setting": ", ".join(sorted({str(r.get("defense")) for r in chosen.values() if r.get("verdict") != "N/A"}))
           or "-",
           "acc0": main0, "acc1": main1,
           "delta": None if main0 is None or main1 is None else main1 - main0,
           "runtime": num((status_entry or {}).get("runtime_min"))}
    for attack, col in (("label_leakage", "leak_acc"), ("villain", "villain_cda"),
                        ("poison_client", "pc_cda"), ("poison_server", "ps_cda")):
        r = chosen.get(attack)
        out[col] = None if r is None or r.get("verdict") == "N/A" else num(r.get("accuracy_with_defense"))
    return out


def read_log_lines(key, model, cut):
    path = log_path(key, model, cut)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [ANSI_RE.sub("", ln.split("\r")[-1]).rstrip() for ln in f.read().split("\n")]
    return [ln for ln in lines if ln.strip() and "it/s]" not in ln and "s/it]" not in ln and "%|" not in ln]


def is_separator(line):
    t = line.strip()
    return len(t) >= 10 and set(t) <= set("=-#")


def log_progress(key, model, cut):
    lines = read_log_lines(key, model, cut)
    if lines is None:
        return None, False
    keep = [False] * len(lines)
    for i, ln in enumerate(lines):
        m = ITER_RE.search(ln)
        if m:
            keep[i] = m.group(1) == m.group(2)
            continue
        if EPOCH_RE.search(ln) or CONTEXT_RE.match(ln):
            keep[i] = True
        if i > 0 and is_separator(lines[i - 1]) and not is_separator(ln):
            keep[i] = True
        if "RESULTS" in ln:
            seps, j = 0, i + 1
            keep[max(0, i - 1)] = keep[i] = True
            while j < len(lines) and j < i + 40 and seps < 2:
                keep[j] = True
                seps += is_separator(lines[j])
                j += 1
    out = []
    for ln, k in zip(lines, keep):
        if k and not (is_separator(ln) and out and is_separator(out[-1])):
            out.append(ln)
    truncated = len(out) > PROGRESS_LINES_PER_RUN
    return out[:PROGRESS_LINES_PER_RUN], truncated


def runner_console_summary(key, model, cut):
    lines = read_log_lines(key, model, cut)
    if lines is None:
        return None, None
    starts = [i for i, ln in enumerate(lines) if any(m in ln for m in LOG_MARKERS)]
    summary = lines[max(0, starts[0] - 2):] if starts else []
    return summary[-600:], lines[-80:]


def banner(text):
    print("\n" + "#" * 94 + f"\n#   {text}\n" + "#" * 94)


def print_console_tables(runs, keys, status):
    for model, cut in runs:
        for key in keys:
            rows = load_rows(key, model, cut)
            print("\n" + "=" * 94)
            print(f"   ALL ATTACKS vs {DEFENSES[key][0].upper()} SUMMARY -- {model} on {DATASET} (cut {cut})")
            print("=" * 94)
            print(f"{'Attack':<20} {'PSNR (dB)':>11} {'SSIM':>9} {'MSE':>9} {'Leak AUC':>10} {'LIA (%)':>9} "
                  f"{'ASR (%)':>9} {'CDA (%)':>9}")
            print("-" * 94)
            for t in attack_table(key, rows):
                c = screenshot_cells(t)
                print(f"  {t['label']:<18} {c[0]:>11} {c[1]:>9} {c[2]:>9} {c[3]:>10} {c[4]:>9} {c[5]:>9} {c[6]:>9}")
            print("=" * 94)
            detail = leakage_detail(key, rows)
            if detail and any(num(v) is not None for _, v in detail):
                print(f"   LABEL LEAKAGE DETAIL (95% quantile leak AUC over batches) -- {DEFENSES[key][0].upper()}")
                for name, v in detail:
                    print(f"  {name:<28} {fmt(v, '.4f'):>10}")
        print("\n" + "=" * 128)
        print(f"   ACCURACY RESULTS -- {model} on {DATASET} (cut {cut})")
        print("=" * 128)
        print(f"  {'Defense':<18} {'Acc no def (%)':>15} {'Acc defended (%)':>17} {'dAcc':>8} {'LeakTask (%)':>13} "
              f"{'VILLAIN CDA':>12} {'BD-C CDA':>9} {'BD-S CDA':>9} {'Time (min)':>11}")
        print("-" * 128)
        for key in keys:
            a = accuracy_row(key, load_rows(key, model, cut), status.get(status_key(model, cut, key)))
            print(f"  {a['defense']:<18} {fmt(a['acc0'], '.2f'):>15} {fmt(a['acc1'], '.2f'):>17} "
                  f"{fmt(a['delta'], '+.2f'):>8} {fmt(a['leak_acc'], '.2f'):>13} {fmt(a['villain_cda'], '.2f'):>12} "
                  f"{fmt(a['pc_cda'], '.2f'):>9} {fmt(a['ps_cda'], '.2f'):>9} {fmt(a['runtime'], '.1f'):>11}")
        print("=" * 128)


def build_pdf(runs, keys, status, path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
                                    Preformatted)

    styles = getSampleStyleSheet()
    h1, h2, h3, body = styles["Title"], styles["Heading1"], styles["Heading2"], styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=9)
    cell = ParagraphStyle("cell", parent=body, fontSize=7, leading=8.5)
    tiny = ParagraphStyle("tiny", parent=body, fontSize=6.3, leading=7.6)
    mono = ParagraphStyle("mono", parent=body, fontName="Courier", fontSize=6.0, leading=7.0)
    grid = [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9ecef")),
            ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]

    def P(text, style=cell):
        return Paragraph(escape(clean_text(text)), style)

    def shade(v, c, r):
        return [("BACKGROUND", (c, r), (c, r), colors.HexColor(VERDICT_COLORS[v]))] if v in VERDICT_COLORS else []

    def wrap_mono(lines, width=212):
        out = []
        for ln in lines:
            while len(ln) > width:
                out.append(ln[:width])
                ln = "    " + ln[width:]
            out.append(ln)
        return "\n".join(out)

    legend = Paragraph(
        "Shaded = headline metric of the attack, coloured by the runner's verdict: "
        "<font backColor='#d4edda'>&nbsp;SUCCESS&nbsp;</font> defense works, "
        "<font backColor='#fff3cd'>&nbsp;PARTIAL&nbsp;</font>, "
        "<font backColor='#f4c7c3'>&nbsp;FAILED&nbsp;</font> attack still works, "
        "<font backColor='#e2e3e5'>&nbsp;ATTACK-WEAK&nbsp;</font> attack weak even without defense. "
        f"{DASH} = column does not apply to this attack. N/A = the runner marks the attack not applicable to this "
        "defense's threat model (reason in the verdict table). NOT RUN = no result yet (pending or crashed).", small)

    total = len(runs) * len(keys)
    states = [status.get(status_key(m, c, k), {}).get("status", "pending") for m, c in runs for k in keys]
    story = [Paragraph("Split Learning Defenses vs. All Attacks \u2014 Full Results", h1),
             Paragraph(f"Dataset <b>{DATASET}</b> &nbsp;|&nbsp; Models <b>{', '.join(dict.fromkeys(m for m, _ in runs))}"
                       f"</b> &nbsp;|&nbsp; Cut layers <b>{', '.join(str(c) for c in dict.fromkeys(c for _, c in runs))}"
                       f"</b> &nbsp;|&nbsp; Report built {now()}", body),
             Paragraph(f"Progress: <b>{states.count('done')}</b> of {total} runs done, "
                       f"<b>{states.count('failed')}</b> failed, <b>{states.count('running')}</b> running, "
                       f"<b>{total - states.count('done') - states.count('failed') - states.count('running')}</b> "
                       "pending. This PDF is rebuilt after every run, so it always shows everything finished so far.",
                       body), Spacer(1, 3 * mm)]

    settings = [["Setting", "Value"], ["Dataset", DATASET], ["Models", ", ".join(MODELS)],
                ["Cut layers", ", ".join(map(str, CUT_LAYERS))],
                ["Defenses", ", ".join(DEFENSES[k][0] for k in keys)],
                ["Attacks (run by every runner)", ", ".join(ATTACK_LABEL[a] for a in ATTACK_ORDER)],
                ["Quick test", str(QUICK_TEST)], ["Runner FSHA fixes applied", str(APPLY_RUNNER_FIXES)],
                ["Checkpoint preparation", str(PREPARE_CHECKPOINTS)],
                ["Runner overrides", json.dumps(RUNNER_OVERRIDES) if RUNNER_OVERRIDES else "none"]]
    for k in keys:
        settings.append([f"{DEFENSES[k][0]} setting in main tables", ", ".join(DEFENSES[k][3])])
    t = Table([settings[0]] + [[P(a, small), P(b, small)] for a, b in settings[1:]], colWidths=[70 * mm, 190 * mm],
              hAlign="LEFT")
    t.setStyle(TableStyle(grid + [("ALIGN", (0, 0), (-1, -1), "LEFT")]))
    story += [t, Spacer(1, 3 * mm), Paragraph("How to read this report", h3), Paragraph(
        "Section 1 = run status. Section 2 = master summary (every model, cut layer, defense and attack on one "
        "table). Section 3 = per model and cut layer: accuracy table and a defense-vs-attack verdict matrix. "
        "Section 4 = one page per defense in the attack-table layout (PSNR, SSIM, MSE, Leak AUC, LIA, ASR, CDA), "
        "with the label-leakage detail and the before/after verdicts. Appendix A = every setting each runner "
        "swept. Appendix B = every field of every result row (A to Z). Appendix C = each runner's own printed "
        "summary tables copied from its log, and the end of the log for failed runs. Appendix D = every "
        "training epoch line and every attack RESULTS block from the logs. All numbers come from your "
        "defense runners. Leak AUC = max of the norm and cosine attacks (95% quantile over batches). For "
        "SplitGuard (detection only) the FSHA row shows reconstruction quality if the client stops at detection.",
        small)]

    story += [PageBreak(), Paragraph("1. Run status", h2)]
    data = [["Model", "Cut", "Defense", "Status", "Exit code", "Rows", "Runtime (min)", "Started", "Finished",
             "Log file"]]
    sh = []
    for i, (m, c, k) in enumerate([(m, c, k) for m, c in runs for k in keys], start=1):
        s = status.get(status_key(m, c, k), {})
        st = s.get("status", "pending")
        data.append([m, str(c), DEFENSES[k][0], st, text_or_dash(s.get("exit_code")), text_or_dash(s.get("rows")),
                     fmt(s.get("runtime_min"), ".1f"), s.get("started", DASH), s.get("finished", DASH),
                     P(os.path.relpath(s["log"], ROOT) if s.get("log") else DASH, tiny)])
        sh += shade({"done": "SUCCESS", "failed": "FAILED", "running": "PARTIAL"}.get(st, ""), 3, i)
    t = Table(data, colWidths=[22 * mm, 10 * mm, 28 * mm, 20 * mm, 16 * mm, 12 * mm, 20 * mm, 27 * mm, 27 * mm,
                               91 * mm], repeatRows=1)
    t.setStyle(TableStyle(grid + sh + [("FONTSIZE", (0, 1), (-1, -1), 7.5)]))
    story.append(t)

    story += [PageBreak(), Paragraph("2. Master summary table (all models, cut layers, defenses and attacks)", h2),
              legend, Spacer(1, 2 * mm)]
    data = [["Model", "Cut", "Defense", "Attack", "PSNR (dB)", "SSIM", "MSE", "Leak AUC", "LIA (%)", "ASR (%)",
             "CDA (%)", "Verdict", "Works?"]]
    sh = []
    for m, c in runs:
        for k in keys:
            for t_ in attack_table(k, load_rows(k, m, c)):
                cells = screenshot_cells(t_)
                r = t_["row"] or {}
                data.append([m, str(c), DEFENSES[k][0], t_["label"], *cells, t_["verdict"] or DASH,
                             text_or_dash(r.get("defense_working"))])
                i = len(data) - 1
                sh += shade(t_["verdict"], 11, i)
    t = Table(data, colWidths=[22 * mm, 9 * mm, 27 * mm, 27 * mm] + [18 * mm] * 7 + [21 * mm, 22 * mm],
              repeatRows=1)
    t.setStyle(TableStyle(grid + sh + [("FONTSIZE", (0, 1), (-1, -1), 7.2), ("TOPPADDING", (0, 1), (-1, -1), 1.5),
                                       ("BOTTOMPADDING", (0, 1), (-1, -1), 1.5)]))
    story.append(t)

    for m, c in runs:
        story += [PageBreak(), Paragraph(f"3. {m} \u2014 cut layer {c} \u2014 accuracy results", h2)]
        acc = [["Defense", "Setting", "Acc. no\ndefense (%)", "Acc. with\ndefense (%)", "Acc change\n(pts)",
                "Label-leak\ntask acc (%)", "VILLAIN\nCDA (%)", "Backdoor(C)\nCDA (%)", "Backdoor(S)\nCDA (%)",
                "Runtime\n(min)"]]
        for k in keys:
            a = accuracy_row(k, load_rows(k, m, c), status.get(status_key(m, c, k)))
            acc.append([a["defense"], P(a["setting"]), fmt(a["acc0"], ".2f"), fmt(a["acc1"], ".2f"),
                        fmt(a["delta"], "+.2f"), fmt(a["leak_acc"], ".2f"), fmt(a["villain_cda"], ".2f"),
                        fmt(a["pc_cda"], ".2f"), fmt(a["ps_cda"], ".2f"), fmt(a["runtime"], ".0f")])
        t = Table(acc, colWidths=[30 * mm, 45 * mm] + [24 * mm] * 8, repeatRows=1)
        t.setStyle(TableStyle(grid))
        story += [t, Spacer(1, 2 * mm), Paragraph(
            "Main-task accuracy (10-class) comes from the runner's reconstruction rows (no defense = undefended "
            "model, with defense = defended model). Label-leak task = the binary task of the label-leakage attack. "
            f"CDA = clean-data accuracy after the backdoor attack. {DASH} = not reported by that runner.", small),
            Spacer(1, 5 * mm), Paragraph(f"{m} \u2014 cut {c} \u2014 does the defense work?", h3)]
        matrix = [["Defense"] + [ATTACK_LABEL[a] for a in ATTACK_ORDER]]
        sh = []
        for i, k in enumerate(keys, start=1):
            chosen = pick_rows(k, load_rows(k, m, c))
            line = [DEFENSES[k][0]]
            for j, a in enumerate(ATTACK_ORDER, start=1):
                r = chosen.get(a)
                line.append(P("not run" if r is None else (r.get("defense_working") or r.get("verdict") or "?")))
                sh += shade(str(r.get("verdict")) if r else "", j, i)
            matrix.append(line)
        t = Table(matrix, colWidths=[30 * mm] + [30 * mm] * 8, repeatRows=1)
        t.setStyle(TableStyle(grid + sh))
        story.append(t)

    for m, c in runs:
        for k in keys:
            rows = load_rows(k, m, c)
            s = status.get(status_key(m, c, k), {})
            story += [PageBreak(), Paragraph(f"4. {m} \u2014 cut {c} \u2014 all attacks vs {DEFENSES[k][0]}", h2),
                      Paragraph(f"Run status: <b>{s.get('status', 'pending')}</b> &nbsp;|&nbsp; runtime "
                                f"{fmt(s.get('runtime_min'), '.1f')} min &nbsp;|&nbsp; finished "
                                f"{s.get('finished', DASH)}", small), Spacer(1, 2 * mm)]
            if not rows:
                story.append(Paragraph("No results for this run yet (pending, or the runner crashed before writing "
                                       "its CSV -- see Appendix C and the log file).", body))
                continue
            table = attack_table(k, rows)
            data = [["Attack", "PSNR (dB)", "SSIM", "MSE", "Leak AUC", "LIA (%)", "ASR (%)", "CDA (%)"]]
            sh = []
            for i, tr in enumerate(table, start=1):
                data.append([tr["label"], *screenshot_cells(tr)])
                col = 2 if tr["attack"] in RECON else 4 if tr["attack"] == "label_leakage" else 6
                sh += shade(tr["verdict"], col, i)
            main = Table(data, colWidths=[34 * mm] + [22 * mm] * 7, repeatRows=1, hAlign="LEFT")
            main.setStyle(TableStyle(grid + sh))
            side = ""
            detail = leakage_detail(k, rows)
            if detail and any(num(v) is not None for _, v in detail):
                d = [[Paragraph("<b>LABEL LEAKAGE DETAIL</b><br/>(95% quantile leak AUC over batches)", cell), ""]]
                d += [[n, fmt(v, ".4f")] for n, v in detail]
                side = Table(d, colWidths=[50 * mm, 18 * mm])
                side.setStyle(TableStyle(grid + [("SPAN", (0, 0), (1, 0)), ("ALIGN", (0, 0), (0, -1), "LEFT")]))
            pair = Table([[main, side]], colWidths=[192 * mm, 76 * mm], hAlign="LEFT")
            pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
            story += [pair, Spacer(1, 2 * mm), legend]

            v = [["Attack", "Setting", "Metric", "No\ndefense", "With\ndefense", "Acc0\n(%)", "Acc\n(%)", "Verdict",
                  "Working", "Reason"]]
            sh = []
            for i, tr in enumerate(table, start=1):
                r = tr["row"] or {}
                spec = ".0f" if str(r.get("metric")) == "Detect@" else ".2f" if tr["attack"] in BACKDOOR else ".4f"
                v.append([tr["label"], P(r.get("defense") or DASH), text_or_dash(r.get("metric")),
                          fmt(r.get("metric_no_defense"), spec, text_or_dash(r.get("metric_no_defense"))),
                          fmt(r.get("metric_with_defense"), spec, text_or_dash(r.get("metric_with_defense"))),
                          fmt(r.get("accuracy_no_defense"), ".2f"), fmt(r.get("accuracy_with_defense"), ".2f"),
                          tr["verdict"] or DASH, text_or_dash(r.get("defense_working")),
                          P(clean_text(r.get("reason"))[:420])])
                sh += shade(tr["verdict"], 7, i)
            t = Table(v, colWidths=[24 * mm, 24 * mm, 14 * mm, 17 * mm, 18 * mm, 13 * mm, 13 * mm, 18 * mm,
                                    20 * mm, 106 * mm], repeatRows=1)
            t.setStyle(TableStyle(grid + sh + [("FONTSIZE", (0, 1), (-1, -1), 7)]))
            story += [Spacer(1, 4 * mm), Paragraph("Verdicts (before -> after, from the runner)", h3), t]

    story += [PageBreak(), Paragraph("Appendix A \u2014 every setting each runner swept", h2),
              Paragraph("Cell = metric with defense for that setting (SSIM / leak AUC / ASR % / detection batch). "
                        "N/A = not applicable by the runner's threat model.", small)]
    for m, c in runs:
        for k in keys:
            rows = [r for r in load_rows(k, m, c) if r.get("verdict") != "BASELINE"]
            names = list(dict.fromkeys(str(r.get("defense")) for r in rows))
            if not names:
                continue
            data = [["Setting"] + [ATTACK_LABEL[a] for a in ATTACK_ORDER]]
            for sname in names:
                line = [P(sname)]
                for a in ATTACK_ORDER:
                    r = next((x for x in rows if str(x.get("defense")) == sname and x.get("attack") == a), None)
                    if r is None:
                        line.append("")
                    elif r.get("verdict") == "N/A":
                        line.append("N/A")
                    else:
                        spec = ".0f" if str(r.get("metric")) == "Detect@" else ".2f" if a in BACKDOOR else ".4f"
                        line.append(fmt(r.get("metric_with_defense"), spec, text_or_dash(r.get("metric_with_defense"))))
                data.append(line)
            t = Table(data, colWidths=[40 * mm] + [28 * mm] * 8, repeatRows=1)
            t.setStyle(TableStyle(grid + [("FONTSIZE", (0, 1), (-1, -1), 7.5)]))
            story += [Spacer(1, 4 * mm), Paragraph(f"{m} \u2014 cut {c} \u2014 {DEFENSES[k][0]}", h3), t]

    story += [PageBreak(), Paragraph("Appendix B \u2014 every field of every result row", h2),
              Paragraph("One line per row the runner wrote (including its no-defense baseline rows). Empty fields "
                        "are left out.", small)]
    skip = {"defense_key", "defense_name", "exit_code", "runtime_min"}
    for m, c in runs:
        for k in keys:
            rows = load_rows(k, m, c)
            if not rows:
                continue
            data = [["#", "Attack / setting", "All fields"]]
            for i, r in enumerate(rows, start=1):
                fields = [f"<b>{escape(str(key_))}</b>={escape(clean_text(val))[:300]}" for key_, val in r.items()
                          if key_ not in skip and clean_text(val) != ""]
                data.append([str(i), P(f"{r.get('attack')} / {r.get('defense')}"),
                             Paragraph("; ".join(fields), tiny)])
            t = Table(data, colWidths=[8 * mm, 38 * mm, 227 * mm], repeatRows=1)
            t.setStyle(TableStyle(grid + [("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                                          ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#cccccc"))]))
            story += [Spacer(1, 4 * mm), Paragraph(f"{m} \u2014 cut {c} \u2014 {DEFENSES[k][0]}", h3), t]

    story += [PageBreak(), Paragraph("Appendix C \u2014 runner console output (from the logs)", h2),
              Paragraph("The summary tables each runner printed at the end, copied from its log. For runs that "
                        "failed, the last lines of the log are shown so the error is visible here.", small)]
    for m, c in runs:
        for k in keys:
            summary, tail = runner_console_summary(k, m, c)
            if summary is None:
                continue
            s = status.get(status_key(m, c, k), {})
            story += [Spacer(1, 4 * mm), Paragraph(f"{m} \u2014 cut {c} \u2014 {DEFENSES[k][0]} "
                                                   f"({s.get('status', 'pending')})", h3)]
            if summary:
                story.append(Preformatted(wrap_mono(summary), mono))
            if s.get("status") != "done" and tail:
                story += [Paragraph("End of log:", small), Preformatted(wrap_mono(tail), mono)]
            if not summary and s.get("status") == "done":
                story.append(Paragraph("No summary section found in the log.", small))

    story += [PageBreak(), Paragraph("Appendix D \u2014 training progress and attack result blocks (from the logs)", h2),
              Paragraph("Every per-epoch line the runner printed (AE decoder Val MSE, FSHA Critic / Hijack / "
                        "Pilot-Recon, label-leakage, VILLAIN phases, backdoor and defense training epochs), every "
                        "RESULTS block (MSE / PSNR / SSIM, ASR, CDA, LIA ...), the headers that say which attack and "
                        "whether it is the baseline or the defended run, and the last white-box iteration per image. "
                        "Progress bars are left out. The complete output of each run is in its log file "
                        "(results/all_defenses/logs).", small)]
    for m, c in runs:
        for k in keys:
            progress, truncated = log_progress(k, m, c)
            if progress is None:
                continue
            s = status.get(status_key(m, c, k), {})
            story += [Spacer(1, 4 * mm), Paragraph(f"{m} \u2014 cut {c} \u2014 {DEFENSES[k][0]} "
                                                   f"({s.get('status', 'pending')})", h3)]
            if progress:
                story.append(Preformatted(wrap_mono(progress), mono))
            else:
                story.append(Paragraph("No epoch or result lines found in the log.", small))
            if truncated:
                story.append(Paragraph(f"Cut at {PROGRESS_LINES_PER_RUN} lines -- see "
                                       f"{escape(os.path.relpath(log_path(k, m, c), ROOT))} for the rest.", small))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.drawString(12 * mm, 6 * mm, f"All defenses vs all attacks -- {DATASET} -- built {now()}")
        canvas.drawRightString(landscape(A4)[0] - 12 * mm, 6 * mm, f"page {doc.page}")
        canvas.restoreState()

    tmp = path + ".tmp"
    doc = SimpleDocTemplate(tmp, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title=f"Defenses vs attacks ({DATASET})")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    os.replace(tmp, path)


def build_report(runs, keys, status, console=False):
    import pandas as pd
    frames = [read_csv(snapshot_path(k, m, c)) for m, c in runs for k in keys if os.path.exists(snapshot_path(k, m, c))]
    combined = os.path.join(out_dir(), f"all_defenses_results_{DATASET}.csv")
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(combined, index=False)
    table_rows = []
    for m, c in runs:
        for k in keys:
            for t in attack_table(k, load_rows(k, m, c)):
                table_rows.append({"model": m, "cut_layer": c, "defense": DEFENSES[k][0], "attack": t["label"],
                                   "setting": (t["row"] or {}).get("defense"), "psnr": t["psnr"], "ssim": t["ssim"],
                                   "mse": t["mse"], "leak_auc": t["leak"], "lia": t["lia"], "asr": t["asr"],
                                   "cda": t["cda"], "verdict": t["verdict"]})
    summary = os.path.join(out_dir(), f"all_defenses_summary_table_{DATASET}.csv")
    pd.DataFrame(table_rows).to_csv(summary, index=False)
    pdf = os.path.join(out_dir(), f"all_defenses_report_{DATASET}.pdf")
    try:
        build_pdf(runs, keys, status, pdf)
        print(f"\n[run_all_defenses] PDF report updated -> {pdf}")
    except ImportError:
        print("\n[!] reportlab is not installed: pip install reportlab   (then rerun with --report-only)")
    except Exception as exc:
        print(f"\n[!] PDF build failed ({type(exc).__name__}: {exc}); CSV results are still saved.")
    if console:
        print_console_tables(runs, keys, status)
        print(f"\n  PDF report      -> {pdf}")
        print(f"  Summary table   -> {summary}")
        print(f"  All raw rows    -> {combined}")
        print(f"  Logs            -> {os.path.join(out_dir(), 'logs')}")


def main():
    global DATASET, MODELS, CUT_LAYERS, DEFENSES_TO_RUN, QUICK_TEST, RESUME, PREPARE_CHECKPOINTS
    p = argparse.ArgumentParser(description="Run every defense runner against all attacks and build the PDF report.")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()
    QUICK_TEST = QUICK_TEST or a.quick
    RESUME = RESUME and not a.no_resume

    from config import Config
    Config.DATASET = DATASET
    runs = [(m, c) for m in MODELS for c in CUT_LAYERS]
    keys = [k for k in DEFENSES_TO_RUN if k in DEFENSES]
    status = load_status()
    started = time.time()

    print("=" * 94)
    print("  ALL DEFENSES vs ALL ATTACKS")
    print(f"  Dataset    : {DATASET}\n  Models     : {MODELS}\n  Cut layers : {CUT_LAYERS}")
    print(f"  Defenses   : {[DEFENSES[k][0] for k in keys]}")
    print(f"  Attacks    : {[ATTACK_LABEL[x] for x in ATTACK_ORDER]}")
    print(f"  Runs       : {len(runs) * len(keys)} (models x cut layers x defenses)")
    print(f"  Mode       : {'report only' if a.report_only else 'QUICK TEST' if QUICK_TEST else 'full run'}"
          f" | resume {RESUME}")
    print(f"  PDF        : {os.path.join(out_dir(), f'all_defenses_report_{DATASET}.pdf')} (updated after every run)")
    print("=" * 94)

    if not a.report_only:
        build_report(runs, keys, status)
        for model, cut in runs:
            todo = [k for k in keys if not (RESUME and status.get(status_key(model, cut, k), {}).get("status") == "done"
                                            and load_rows(k, model, cut))]
            for k in keys:
                if k not in todo:
                    print(f"[resume] {DEFENSES[k][0]} / {model} / cut {cut} already done -- skipping")
            if not todo:
                continue
            if PREPARE_CHECKPOINTS:
                banner(f"PREPARE CHECKPOINTS  |  {model}  |  cut {cut}  |  {DATASET}")
                code, _ = run_child(base_job(model, cut, "prepare"),
                                    os.path.join(out_dir(), "logs", f"prepare_{model.lower()}_cut{cut}_{DATASET}.log"))
                if code != 0:
                    print(f"[!] prepare step failed for {model} cut {cut} (exit {code}); runners will mark "
                          "missing checkpoints as N/A")
            for k in todo:
                run_defense(k, model, cut, status)
                build_report(runs, keys, status)

    build_report(runs, keys, status, console=True)
    failed = [f"{DEFENSES[k][0]}/{m}/cut{c}" for m, c in runs for k in keys
              if status.get(status_key(m, c, k), {}).get("status") == "failed"]
    if failed:
        print(f"\n  [!] runs that exited with an error (partial results kept): {', '.join(failed)}")
    print(f"\n  Total time: {(time.time() - started) / 3600:.2f} h")


if __name__ == "__main__":
    main()