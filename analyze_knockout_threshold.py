"""
analyze_knockout_threshold.py — Same rigor as analyze_draw_gap.py but for knockout.

Uses the same methodology applied to groups:
  - Sweep theta_D_knockout on the calibration set (WC knockout 1986-2018, 144 matches)
  - Evaluate confusion matrices on the validation set (WC 2022 knockout, 16 matches)
  - Check how many val matches fall in the critical band around theta=0.28
  - Segment draw FN by |elo_diff|

The 16-match validation is small — conclusions will be weaker than groups.
That's expected and should be declared, not hidden.

Run:
    py analyze_knockout_threshold.py
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.data.load import load_results
from src.features.elo import compute_elo
from src.features.h2h import compute_h2h
from src.features.tournament import add_tournament_features

CONFIG_PATH = "models/v5_config.json"
MODEL_PATH  = "models/poisson_dc_v5.joblib"
CLASSES     = ["H", "D", "A"]
VAL_START   = pd.Timestamp("2022-01-01")
TEST_START  = pd.Timestamp("2023-01-01")


def label_result(hs, as_):
    if hs > as_: return "H"
    if hs < as_: return "A"
    return "D"


def apply_threshold(proba, t):
    return np.where(proba[:, 1] > t, "D", np.where(proba[:, 0] >= proba[:, 2], "H", "A"))


def sweep(proba, y, lo=0.15, hi=0.55, step=0.01, metric="f1_macro"):
    best_val, best_t = -1.0, lo
    for t in np.round(np.arange(lo, hi + step, step), 2):
        pred = apply_threshold(proba, t)
        if metric == "f1_macro":
            v = f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)
        elif metric == "f1_draw":
            v = f1_score(y, pred, labels=["D"], average="macro", zero_division=0)
        else:  # draw_recall
            tp = ((pred == "D") & (y == "D")).sum()
            fn = ((pred != "D") & (y == "D")).sum()
            v  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if v > best_val:
            best_val, best_t = v, float(t)
    return best_t, best_val


def draw_stats(pred, y):
    tp = int(((pred == "D") & (y == "D")).sum())
    fp = int(((pred == "D") & (y != "D")).sum())
    fn = int(((pred != "D") & (y == "D")).sum())
    tn = int(((pred != "D") & (y != "D")).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, prec=prec, rec=rec, f1=f1)


def print_block(label, proba, y, theta):
    pred = apply_threshold(proba, theta)
    acc  = (pred == y).mean()
    f1m  = f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)
    ds   = draw_stats(pred, y)
    dr_p = (pred == "D").mean()
    dr_r = (y == "D").mean()
    print(f"  [{label}]  theta={theta:.2f}")
    print(f"    acc={acc:.3f}   F1-macro={f1m:.3f}   "
          f"draw pred {dr_p:.1%} vs real {dr_r:.1%}")
    print(f"    Draw:  TP={ds['tp']}  FP={ds['fp']}  FN={ds['fn']}  TN={ds['tn']}  "
          f"prec={ds['prec']:.2f}  rec={ds['rec']:.2f}  F1={ds['f1']:.2f}")
    return pred


# ── load ──────────────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
model = joblib.load(MODEL_PATH)
p, hp = cfg["priors"], cfg["hyperparams"]

df_raw = load_results()
df, _  = compute_elo(df_raw, home_advantage=cfg["home_advantage"])
df["result"] = df.apply(lambda r: label_result(r.home_score, r.away_score), axis=1)
df = compute_h2h(df, p["global_home_avg"], p["global_away_avg"], k=hp["h2h_k"])
df = add_tournament_features(df)

# ── splits ────────────────────────────────────────────────────────────────────
wc_ko_train = df[
    (df["is_world_cup"] == 1) &
    (df["is_knockout"] == 1) &
    (df["date"] < VAL_START)
].copy().reset_index(drop=True)

wc_ko_val = df[
    (df["is_world_cup"] == 1) &
    (df["is_knockout"] == 1) &
    (df["date"] >= VAL_START) &
    (df["date"] < TEST_START)
].copy().reset_index(drop=True)

print(f"Calibration set (WC knockout 1986-2018): {len(wc_ko_train)} matches  "
      f"draw rate {(wc_ko_train['result']=='D').mean():.1%}")
print(f"Validation set  (WC 2022 knockout):      {len(wc_ko_val)} matches  "
      f"draw rate {(wc_ko_val['result']=='D').mean():.1%}")
print(f"  NOTE: n=16 — all conclusions carry high uncertainty.")
print()

p_train = model.predict_proba(wc_ko_train)
p_val   = model.predict_proba(wc_ko_val)
y_train = wc_ko_train["result"].values
y_val   = wc_ko_val["result"].values

# ── P(D) distribution — critical band analysis ─────────────────────────────────
print("=" * 62)
print("  P(D) distribution on validation set (16 matches)")
print("  How much mass falls near theta=0.28?")
print("=" * 62)
pd_vals = p_val[:, 1]
bands = [(0.20, 0.24), (0.24, 0.26), (0.26, 0.28), (0.28, 0.30), (0.30, 0.35)]
print(f"\n  P(D) range: min={pd_vals.min():.3f}  mean={pd_vals.mean():.3f}  "
      f"max={pd_vals.max():.3f}  median={np.median(pd_vals):.3f}")
print()
print(f"  {'Band':<14}  {'n matches':>10}  {'% of val':>9}  actual draws in band")
print("  " + "-" * 55)
for lo, hi in bands:
    mask    = (pd_vals >= lo) & (pd_vals < hi)
    n       = mask.sum()
    n_draws = (y_val[mask] == "D").sum()
    pct     = n / len(pd_vals)
    print(f"  [{lo:.2f}, {hi:.2f})    {n:>10}  {pct:>9.1%}  {n_draws} draws")

# Exactly how many are in [0.25, 0.31] — the "critical band" around theta=0.28
crit_mask = (pd_vals >= 0.25) & (pd_vals <= 0.31)
print(f"\n  Critical band [0.25, 0.31] (theta +/- 0.03): "
      f"{crit_mask.sum()} of {len(pd_vals)} matches ({crit_mask.mean():.1%})")
print(f"  => theta=0.28 is decisive for these {crit_mask.sum()} match(es)")

# ── sweep on calibration set ──────────────────────────────────────────────────
print()
print("=" * 62)
print("  Threshold sweep on calibration set (144 matches, 1986-2018)")
print("=" * 62)
theta_f1m, best_f1m = sweep(p_train, y_train, metric="f1_macro")
theta_f1d, best_f1d = sweep(p_train, y_train, metric="f1_draw")
theta_rec, best_rec = sweep(p_train, y_train, metric="draw_recall")
print(f"  f1_macro    -> theta={theta_f1m:.2f}   best={best_f1m:.4f}")
print(f"  f1_draw     -> theta={theta_f1d:.2f}   best={best_f1d:.4f}")
print(f"  draw_recall -> theta={theta_rec:.2f}   best={best_rec:.4f}  [upper bound]")
print(f"  current     -> theta=0.28  (calibrated on this same set)")
print()

# ── confusion matrices on val (16 matches) ────────────────────────────────────
print("=" * 62)
print("  Confusion matrices on WC 2022 knockout val (16 matches)")
print("  WARNING: n=16. Each match = 6.25% accuracy. High variance.")
print("=" * 62)
pred_f1m = print_block("f1_macro   (sweep on calib)", p_val, y_val, theta_f1m)
print()
pred_cur = print_block("current    (theta=0.28)    ", p_val, y_val, 0.28)
print()
pred_f1d = print_block("f1_draw    (sweep on calib)", p_val, y_val, theta_f1d)
print()
pred_rec = print_block("draw_recall               ", p_val, y_val, theta_rec)
print()

# ── individual match breakdown ─────────────────────────────────────────────────
print("=" * 62)
print("  Per-match breakdown on val (sorted by P(D) descending)")
print("=" * 62)
wc_ko_val["p_draw"] = pd_vals
wc_ko_val["abs_elo_diff"] = wc_ko_val["elo_diff"].abs()
wc_ko_val["pred_cur"] = pred_cur

print(f"\n  {'Date':<12} {'Home':<22} {'Away':<22} "
      f"{'P(D)':>5} {'actual':>6} {'pred':>5} {'elo_diff':>9}")
print("  " + "-" * 83)
for _, r in wc_ko_val.sort_values("p_draw", ascending=False).iterrows():
    flag = " <-- DRAW" if r["result"] == "D" else ""
    hit  = "OK" if r["result"] == r["pred_cur"] else "MISS"
    near = " [near theta]" if 0.25 <= r["p_draw"] <= 0.31 else ""
    print(f"  {str(r['date'])[:10]:<12} {r['home_team']:<22} {r['away_team']:<22} "
          f"{r['p_draw']:>5.3f} {r['result']:>6} {r['pred_cur']:>5}  "
          f"{r['elo_diff']:>+9.0f}  {hit}{flag}{near}")

# ── draw FN segmentation ───────────────────────────────────────────────────────
print()
print("=" * 62)
print("  Draw FN segmentation (current theta=0.28)")
print("=" * 62)
draws    = wc_ko_val[wc_ko_val["result"] == "D"]
fn_draws = draws[draws["pred_cur"] != "D"]
n_d, n_fn = len(draws), len(fn_draws)
print(f"\n  Total draws: {n_d}  |  Missed (FN): {n_fn}  |  Hit (TP): {n_d - n_fn}")

if n_d > 0:
    med_elo = draws["abs_elo_diff"].median()
    print(f"\n  By |elo_diff| split at median={med_elo:.0f}:")
    for tier, mask_fn in [
        ("close  (<=median)", draws["abs_elo_diff"] <= med_elo),
        ("apart  (>median)",  draws["abs_elo_diff"] >  med_elo),
    ]:
        nd  = mask_fn.sum()
        nf  = draws[mask_fn & (draws["pred_cur"] != "D")].shape[0]
        avg = draws[mask_fn]["abs_elo_diff"].mean() if nd > 0 else 0
        print(f"  {tier:<22}  draws={nd}  missed={nf}  "
              f"miss rate={nf/nd:.1%}  avg |elo_diff|={avg:.0f}")

    if n_fn > 0:
        print(f"\n  P(D) of draws:")
        tp_pd = draws[draws["pred_cur"] == "D"]["p_draw"]
        fn_pd = fn_draws["p_draw"]
        if len(tp_pd) > 0:
            print(f"    TP: mean={tp_pd.mean():.3f}  "
                  f"min={tp_pd.min():.3f}  max={tp_pd.max():.3f}")
        print(f"    FN: mean={fn_pd.mean():.3f}  "
              f"min={fn_pd.min():.3f}  max={fn_pd.max():.3f}")

print()
print("=" * 62)
print("  Summary")
print("=" * 62)
print(f"  Calibration set draw rate:   {(y_train=='D').mean():.1%}  (n=144)")
print(f"  Validation set draw rate:    {(y_val=='D').mean():.1%}  (n=16, high variance)")
print(f"  Sweep on calib -> theta_f1m: {theta_f1m:.2f}  (current config: 0.28)")
print(f"  Critical band [0.25,0.31]:   {crit_mask.sum()} of 16 val matches")
print()
print("  Interpretation:")
print("  - If sweep produces same theta as config (0.28): calibration is stable")
print("  - If critical band has many matches: theta IS decisive, not marginal")
print("  - n=16 means each match moves accuracy by 6.25pp — declare this")
