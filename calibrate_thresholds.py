"""
Calibrate two decision thresholds for v5 and save them to v5_config.json.

theta_D         : calibrated on val set (2022) excluding WC knockout.
                  Used for all non-knockout predictions.

theta_D_knockout: calibrated on training-set WC knockout matches (1986-2018).
                  Those 144 matches are in-sample for the Poisson model but
                  acceptable for a scalar threshold — overfitting risk is low.
                  Validated on 2022 WC knockout (16 matches, val set).

Next step (once WC 2026 is complete): retrain v5 including 2022 WC data,
then recalibrate theta_D_knockout on 1986-2022 knockout and validate on
the WC 2026 knockout results.

Run:
    py calibrate_thresholds.py
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

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


def log_loss(y_true, y_proba):
    idx = np.array([CLASSES.index(y) for y in y_true])
    p   = y_proba[np.arange(len(y_true)), idx]
    return float(-np.mean(np.log(np.maximum(p, 1e-15))))


def sweep(proba, y, lo=0.10, hi=0.50, step=0.01, metric="f1_macro"):
    best_val, best_t = -1.0, lo
    for t in np.round(np.arange(lo, hi + step, step), 2):
        pred = apply_threshold(proba, t)
        if metric == "f1_macro":
            v = f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)
        else:
            v = f1_score(y, pred, labels=["D"], average="macro", zero_division=0)
        if v > best_val:
            best_val, best_t = v, float(t)
    return best_t, best_val


def report_block(tag, proba, y, theta):
    pred = apply_threshold(proba, theta)
    ll   = log_loss(y, proba)
    acc  = accuracy_score(y, pred)
    f1m  = f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)
    f1d  = f1_score(y, pred, labels=["D"], average="macro", zero_division=0)
    dr_p = (pred == "D").mean()
    dr_r = (np.array(y) == "D").mean()
    print(f"  {tag}")
    print(f"    theta={theta:.2f}  log-loss={ll:.4f}  acc={acc:.3f}  "
          f"F1-macro={f1m:.3f}  F1-draw={f1d:.3f}")
    print(f"    draw pred {dr_p:.1%}  vs  real {dr_r:.1%}")


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
df = add_tournament_features(df)

# ── splits ────────────────────────────────────────────────────────────────────
val  = df[(df["date"] >= VAL_START) & (df["date"] < TEST_START)].copy()

# theta_D general: val set minus WC knockout
val_general = val[~((val["is_world_cup"] == 1) & (val["is_knockout"] == 1))].copy()

# theta_D_knockout calibration: training set WC knockout, excluding 2022 val
wc_ko_train = df[
    (df["is_world_cup"] == 1) &
    (df["is_knockout"] == 1) &
    (df["date"] < VAL_START)
].copy()

# theta_D_knockout validation: 2022 WC knockout only
wc_ko_2022 = df[
    (df["is_world_cup"] == 1) &
    (df["is_knockout"] == 1) &
    (df["date"] >= VAL_START) &
    (df["date"] < TEST_START)
].copy()

print(f"Splits:")
print(f"  val_general (non-WC-KO val):      {len(val_general):>4} matches")
print(f"  wc_ko_train (1986-2018 knockout):  {len(wc_ko_train):>4} matches")
print(f"  wc_ko_2022  (2022 WC knockout):    {len(wc_ko_2022):>4} matches")
print()

# ── get predictions ───────────────────────────────────────────────────────────
p_gen  = model.predict_proba(val_general)
p_ktr  = model.predict_proba(wc_ko_train)
p_k22  = model.predict_proba(wc_ko_2022)

y_gen  = val_general["result"].values
y_ktr  = wc_ko_train["result"].values
y_k22  = wc_ko_2022["result"].values

# ── calibrate theta_D (general) ───────────────────────────────────────────────
theta_D, best_f1_gen = sweep(p_gen, y_gen, metric="f1_macro")
print("=" * 60)
print("  theta_D (general) calibration")
print("=" * 60)
report_block("val_general @ chosen theta_D:", p_gen, y_gen, theta_D)
print()

# ── calibrate theta_D_knockout ────────────────────────────────────────────────
print("=" * 60)
print("  theta_D_knockout calibration (1986-2018 WC knockout)")
print("=" * 60)

# P(D) distribution of knockout predictions
pD_vals = p_ktr[:, 1]
print(f"  P(D) in knockout train: "
      f"min={pD_vals.min():.3f}  mean={pD_vals.mean():.3f}  max={pD_vals.max():.3f}")

theta_D_ko, best_f1_ko = sweep(p_ktr, y_ktr, lo=0.20, hi=0.55, metric="f1_macro")
_, best_f1d_ko = sweep(p_ktr, y_ktr, lo=0.20, hi=0.55, metric="f1_draw")
print()
report_block(f"wc_ko_train @ chosen theta_D_knockout ({theta_D_ko:.2f}):", p_ktr, y_ktr, theta_D_ko)
print()

# ── validate on 2022 WC knockout ──────────────────────────────────────────────
print("=" * 60)
print("  Validation: 2022 WC knockout (val set, not used for calibration)")
print("=" * 60)
pD_vals_22 = p_k22[:, 1]
print(f"  P(D) in 2022 knockout: "
      f"min={pD_vals_22.min():.3f}  mean={pD_vals_22.mean():.3f}  max={pD_vals_22.max():.3f}")
print()
report_block(f"wc_ko_2022 @ theta_D={theta_D:.2f} (general):", p_k22, y_k22, theta_D)
report_block(f"wc_ko_2022 @ theta_D_knockout={theta_D_ko:.2f}:", p_k22, y_k22, theta_D_ko)
print()

# ── per-round breakdown on validation ────────────────────────────────────────
wc_ko_2022_sorted = wc_ko_2022.sort_values("date").reset_index(drop=True)
# Recompute predictions on the sorted copy so row index matches proba row index
p_k22_sorted = model.predict_proba(wc_ko_2022_sorted)
pred_ko      = apply_threshold(p_k22_sorted, theta_D_ko)
print("  2022 WC knockout by round (approximate, sorted by date):")
for i, row in wc_ko_2022_sorted.iterrows():
    rnd    = "R16" if i < 8 else ("QF" if i < 12 else ("SF" if i < 14 else "Final/3rd"))
    actual = row["result"]
    pred   = pred_ko[i]
    p_d_val = p_k22_sorted[i, 1]
    flag   = "OK" if actual == pred else "MISS"
    print(f"    [{rnd}] {row.home_team} vs {row.away_team}  "
          f"actual={actual} pred={pred} P(D)={p_d_val:.2f}  {flag}")

# ── save to config ────────────────────────────────────────────────────────────
print()
print("=" * 60)
cfg["theta_D"]          = theta_D
cfg["theta_D_knockout"] = theta_D_ko
with open(CONFIG_PATH, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"  Saved to {CONFIG_PATH}")
print(f"    theta_D          = {theta_D:.2f}")
print(f"    theta_D_knockout = {theta_D_ko:.2f}")
print()
