"""
analyze_draw_gap.py — Draw gap diagnostic for WC 2026 group stage.

Steps:
  1. Sweep three threshold objectives on val_general (2022, non-WC-KO):
       f1_macro   — current calibration
       f1_draw    — maximizes F1 on the draw class specifically
       draw_recall — maximizes raw recall on draws (diagnostic upper bound)
  2. Print confusion matrices for all three thresholds on 48 WC 2026 group matches.
  3. Segment draw false-negatives by |elo_diff| and H2H depth to test
     whether draw misses concentrate in close matchups (structural) or spread
     randomly (noise / threshold too high).

Run:
    py analyze_draw_gap.py
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
from src.features.form import compute_form
from src.features.tournament import add_tournament_features

CONFIG_PATH  = "models/v5_config.json"
MODEL_PATH   = "models/poisson_dc_v5.joblib"
CLASSES      = ["H", "D", "A"]
VAL_START    = pd.Timestamp("2022-01-01")
TEST_START   = pd.Timestamp("2023-01-01")
TRAIN_CUTOFF = pd.Timestamp("2026-06-10")


# ── helpers ───────────────────────────────────────────────────────────────────

def label_result(hs, as_):
    if hs > as_: return "H"
    if hs < as_: return "A"
    return "D"


def apply_threshold(proba: np.ndarray, theta_d: float) -> np.ndarray:
    return np.where(
        proba[:, 1] > theta_d, "D",
        np.where(proba[:, 0] >= proba[:, 2], "H", "A"),
    )


def sweep(proba: np.ndarray, y: np.ndarray, lo=0.10, hi=0.50, step=0.01,
          metric="f1_macro") -> tuple[float, float]:
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


def draw_stats(pred: np.ndarray, y: np.ndarray) -> dict:
    tp = int(((pred == "D") & (y == "D")).sum())
    fp = int(((pred == "D") & (y != "D")).sum())
    fn = int(((pred != "D") & (y == "D")).sum())
    tn = int(((pred != "D") & (y != "D")).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, prec=prec, rec=rec, f1=f1)


def print_threshold_block(label: str, proba: np.ndarray, y: np.ndarray,
                          theta: float) -> np.ndarray:
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

p  = cfg["priors"]
hp = cfg["hyperparams"]

df_raw = load_results()
df, _  = compute_elo(df_raw, home_advantage=cfg["home_advantage"])
df["result"] = df.apply(lambda r: label_result(r.home_score, r.away_score), axis=1)
df = compute_h2h(df, p["global_home_avg"], p["global_away_avg"], k=hp["h2h_k"])
df = compute_form(df, p["global_avg"], window=hp["form_window"], k=hp["form_k"])
df = add_tournament_features(df)

# ── splits ────────────────────────────────────────────────────────────────────
val_general = df[
    (df["date"] >= VAL_START) & (df["date"] < TEST_START) &
    ~((df["is_world_cup"] == 1) & (df["is_knockout"] == 1))
].copy()

wc2026 = df[
    (df["date"] > TRAIN_CUTOFF) &
    (df["is_world_cup"] == 1) &
    (df["is_knockout"] == 0) &
    df["home_score"].notna() &
    df["away_score"].notna()
].copy()

y_val   = val_general["result"].values
y_wc26  = wc2026["result"].values
p_val   = model.predict_proba(val_general)
p_wc26  = model.predict_proba(wc2026)

print(f"Val general (sweep set):      {len(val_general):>4} matches   "
      f"draw rate {(y_val == 'D').mean():.1%}")
print(f"WC 2026 group (eval set):     {len(wc2026):>4} matches   "
      f"draw rate {(y_wc26 == 'D').mean():.1%}")
print()

# ── sweep ─────────────────────────────────────────────────────────────────────
theta_f1m, best_f1m = sweep(p_val, y_val, metric="f1_macro")
theta_f1d, best_f1d = sweep(p_val, y_val, metric="f1_draw")
theta_rec, best_rec = sweep(p_val, y_val, metric="draw_recall")

print("=" * 62)
print("  Threshold sweep on val_general (2022)")
print("=" * 62)
print(f"  f1_macro    -> theta={theta_f1m:.2f}   best F1-macro={best_f1m:.4f}")
print(f"  f1_draw     -> theta={theta_f1d:.2f}   best F1-draw={best_f1d:.4f}")
print(f"  draw_recall -> theta={theta_rec:.2f}   best recall={best_rec:.4f}  "
      f"[diagnostic upper bound]")
print()

# ── confusion matrices on WC 2026 ─────────────────────────────────────────────
print("=" * 62)
print("  Confusion matrices — WC 2026 group stage (48 matches)")
print("=" * 62)
pred_f1m = print_threshold_block("f1_macro   ", p_wc26, y_wc26, theta_f1m)
print()
pred_f1d = print_threshold_block("f1_draw    ", p_wc26, y_wc26, theta_f1d)
print()
pred_rec = print_threshold_block("draw_recall", p_wc26, y_wc26, theta_rec)
print()

# ── segmentation of draw false-negatives (f1_macro threshold) ─────────────────
print("=" * 62)
print("  Draw false-negative segmentation  (f1_macro threshold)")
print("  Real draws the model missed — systematic or random?")
print("=" * 62)

wc = wc2026.copy()
wc["p_draw"]       = p_wc26[:, 1]
wc["pred"]         = pred_f1m
wc["abs_elo_diff"] = wc["elo_diff"].abs()

draws    = wc[wc["result"] == "D"].copy()
fn_draws = draws[draws["pred"] != "D"].copy()

n_draws, n_fn = len(draws), len(fn_draws)
print(f"\n  Total draws in WC 2026 group: {n_draws}  |  Missed (FN): {n_fn}  "
      f"|  Hit (TP): {n_draws - n_fn}")
print()

# By |elo_diff| relative to draw median
med_elo = draws["abs_elo_diff"].median()
draws["elo_tier"] = np.where(draws["abs_elo_diff"] <= med_elo, "close  (<=median)", "apart  (>median)")
fn_idx = set(fn_draws.index)

print(f"  By |elo_diff| split at median={med_elo:.0f}:")
print(f"  {'Tier':<22}  {'n draws':>7}  {'missed':>7}  {'miss rate':>10}  "
      f"{'avg |elo_diff|':>15}")
for tier in ["close  (<=median)", "apart  (>median)"]:
    mask = draws["elo_tier"] == tier
    nd   = mask.sum()
    nf   = draws[mask].index.isin(fn_idx).sum()
    avg  = draws[mask]["abs_elo_diff"].mean()
    print(f"  {tier:<22}  {nd:>7}  {nf:>7}  {nf/nd:>10.1%}  {avg:>15.0f}")

print()

# By H2H depth (h2h_n == 0 means no directional history)
print(f"  By H2H depth (h2h_n = directional matches seen before this game):")
for has_h2h, label in [(False, "No H2H history (n=0)"), (True, "Has H2H history (n>=1)")]:
    mask  = (draws["h2h_n"] == 0) if not has_h2h else (draws["h2h_n"] > 0)
    nd    = mask.sum()
    nf    = draws[mask].index.isin(fn_idx).sum()
    rate  = nf / nd if nd > 0 else 0.0
    print(f"  {label:<28}  draws={nd}  missed={nf}  miss rate={rate:.1%}")

print()

# P(D) distribution for hits vs misses
if n_fn > 0:
    print(f"  P(D) distribution:")
    print(f"    TP (correctly called draw):  "
          f"mean={draws[~draws.index.isin(fn_idx)]['p_draw'].mean():.3f}  "
          f"min={draws[~draws.index.isin(fn_idx)]['p_draw'].min():.3f}  "
          f"max={draws[~draws.index.isin(fn_idx)]['p_draw'].max():.3f}")
    print(f"    FN (draws the model missed): "
          f"mean={fn_draws['p_draw'].mean():.3f}  "
          f"min={fn_draws['p_draw'].min():.3f}  "
          f"max={fn_draws['p_draw'].max():.3f}")
    print()

# Individual draw FN — sorted by |elo_diff| ascending (closest matchups first)
print(f"  Individual draws missed (sorted by |elo_diff|, closest first):")
print(f"  {'Date':<12} {'Home':<22} {'Away':<22} {'P(D)':>5} "
      f"{'elo_diff':>9} {'h2h_n':>6} {'pred':>5}")
print("  " + "-" * 83)
for _, row in fn_draws.sort_values("abs_elo_diff").iterrows():
    print(f"  {str(row['date'])[:10]:<12} {row['home_team']:<22} {row['away_team']:<22} "
          f"{row['p_draw']:>5.3f} {row['elo_diff']:>9.0f} {int(row['h2h_n']):>6}  "
          f"{row['pred']:>4}")

print()

# Also print the TP draws for comparison
tp_draws = draws[~draws.index.isin(fn_idx)]
if len(tp_draws) > 0:
    print(f"  Draws correctly predicted (TP):")
    print(f"  {'Date':<12} {'Home':<22} {'Away':<22} {'P(D)':>5} "
          f"{'elo_diff':>9} {'h2h_n':>6}")
    print("  " + "-" * 78)
    for _, row in tp_draws.sort_values("abs_elo_diff").iterrows():
        print(f"  {str(row['date'])[:10]:<12} {row['home_team']:<22} {row['away_team']:<22} "
              f"{row['p_draw']:>5.3f} {row['elo_diff']:>9.0f} {int(row['h2h_n']):>6}")

print()
print("=" * 62)
print("  Interpretation guide")
print("=" * 62)
print("  If FN miss rate >> TP rate for 'close' tier: structural gap")
print("    -> model lacks a 'parity' signal; threshold shift won't fix it")
print("  If FN spread across both tiers with similar |elo_diff|: noise")
print("    -> threshold adjustment (f1_draw) is the right lever")
print("  P(D) of FN << P(D) of TP: model has some ordering ability")
print("    -> calibration (lower theta_D) can recover some draws")
print("  P(D) of FN ~= P(D) of TP: model can't distinguish -> structural")
