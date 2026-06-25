"""
Compare v5 (full history) vs v5 with 1990+ training window.

Elo is always computed on the full dataset — cutting Elo history would
give wrong ratings for teams with long histories.

Only the Poisson model training set is windowed:
  full:    all matches < 2023-01-01
  1990+:   matches >= 1990-01-01 and < 2023-01-01

Val / test splits stay the same (val: 2022, test: 2023+).

Run:
    py compare_window.py
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.data.load import load_results
from src.features.elo import compute_elo
from src.features.h2h import compute_h2h
from src.features.form import compute_form
from src.features.tournament import add_tournament_features, WC_FEATURES
from src.features.poisson import PoissonDC

HOME_ADVANTAGE = 20.0
WINDOW_START   = pd.Timestamp("1990-01-01")
TRAIN_CUTOFF   = pd.Timestamp("2023-01-01")
VAL_START      = pd.Timestamp("2022-01-01")
CLASSES        = ["H", "D", "A"]
H2H_K          = 5.0
FORM_WINDOW    = 5
FORM_K         = 3.0


def label_result(hs, as_):
    if hs > as_: return "H"
    if hs < as_: return "A"
    return "D"


def apply_threshold(proba, theta_d):
    return np.where(
        proba[:, 1] > theta_d, "D",
        np.where(proba[:, 0] >= proba[:, 2], "H", "A"),
    )


def log_loss(y_true, y_proba):
    idx = np.array([CLASSES.index(y) for y in y_true])
    p   = y_proba[np.arange(len(y_true)), idx]
    return float(-np.mean(np.log(np.maximum(p, 1e-15))))


def best_theta(proba, y, lo=0.10, hi=0.45):
    best_f1, best_t = -1.0, lo
    for t in np.round(np.arange(lo, hi + 0.01, 0.01), 2):
        pred = apply_threshold(proba, t)
        f1   = f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def metrics(label, proba, y, theta):
    pred   = apply_threshold(proba, theta)
    return {
        "model":     label,
        "n":         len(y),
        "log_loss":  round(log_loss(y, proba), 4),
        "accuracy":  round(accuracy_score(y, pred), 3),
        "f1_macro":  round(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0), 3),
        "f1_draw":   round(f1_score(y, pred, labels=["D"], average="macro", zero_division=0), 3),
        "draw_pred": round((pred == "D").mean(), 3),
        "draw_act":  round((np.array(y) == "D").mean(), 3),
        "theta_D":   theta,
    }


# ── build features once (Elo on full dataset) ─────────────────────────────────
print("Building features (Elo on full dataset)...")
df_raw = load_results()
df, _  = compute_elo(df_raw, home_advantage=HOME_ADVANTAGE)
df["result"] = df.apply(lambda r: label_result(r.home_score, r.away_score), axis=1)

# val/test masks
val_mask  = (df["date"] >= VAL_START)  & (df["date"] < TRAIN_CUTOFF)
test_mask = df["date"] >= TRAIN_CUTOFF

# ── variant A: full history ───────────────────────────────────────────────────
train_full = df[df["date"] < TRAIN_CUTOFF].copy()
gh_full    = train_full["home_score"].mean()
ga_full    = train_full["away_score"].mean()
gav_full   = (gh_full + ga_full) / 2

df_full = compute_h2h(df.copy(), gh_full, ga_full, k=H2H_K)
df_full = compute_form(df_full, gav_full, window=FORM_WINDOW, k=FORM_K)
df_full = add_tournament_features(df_full)

model_full = PoissonDC(
    extra_home=["h2h_home_goals_mu"] + WC_FEATURES,
    extra_away=["h2h_away_goals_mu"] + WC_FEATURES,
)
model_full.fit(df_full[df_full["date"] < TRAIN_CUTOFF])

# ── variant B: 1990+ window ───────────────────────────────────────────────────
train_win = df[(df["date"] >= WINDOW_START) & (df["date"] < TRAIN_CUTOFF)].copy()
gh_win    = train_win["home_score"].mean()
ga_win    = train_win["away_score"].mean()
gav_win   = (gh_win + ga_win) / 2

df_win = compute_h2h(df.copy(), gh_win, ga_win, k=H2H_K)
df_win = compute_form(df_win, gav_win, window=FORM_WINDOW, k=FORM_K)
df_win = add_tournament_features(df_win)

model_win = PoissonDC(
    extra_home=["h2h_home_goals_mu"] + WC_FEATURES,
    extra_away=["h2h_away_goals_mu"] + WC_FEATURES,
)
model_win.fit(df_win[(df_win["date"] >= WINDOW_START) & (df_win["date"] < TRAIN_CUTOFF)])

print(f"  full train: {len(train_full):,} matches  "
      f"avg goals/match: {gav_full:.3f}")
print(f"  1990+ train: {len(train_win):,} matches  "
      f"avg goals/match: {gav_win:.3f}")
print()

# ── calibrate both on val ─────────────────────────────────────────────────────
val_f   = df_full[val_mask].copy()
val_w   = df_win[val_mask].copy()
y_val   = val_f["result"].values   # same matches, same labels

pf_val  = model_full.predict_proba(val_f)
pw_val  = model_win.predict_proba(val_w)

theta_f, f1_f = best_theta(pf_val, y_val)
theta_w, f1_w = best_theta(pw_val, y_val)

# ── val comparison ────────────────────────────────────────────────────────────
print("=" * 65)
print("  VAL SET (2022)")
print("=" * 65)
rows = pd.DataFrame([
    metrics("full", pf_val, y_val, theta_f),
    metrics("1990+", pw_val, y_val, theta_w),
])
print(rows.to_string(index=False))
print()

# ── test comparison ───────────────────────────────────────────────────────────
test_f  = df_full[test_mask].copy()
test_w  = df_win[test_mask].copy()
y_test  = test_f["result"].values

pf_test = model_full.predict_proba(test_f)
pw_test = model_win.predict_proba(test_w)

print("=" * 65)
print("  TEST SET (2023+, all tournaments)")
print("=" * 65)
rows_t = pd.DataFrame([
    metrics("full",  pf_test, y_test, theta_f),
    metrics("1990+", pw_test, y_test, theta_w),
])
print(rows_t.to_string(index=False))
print()

# ── WC 2026 group stage ───────────────────────────────────────────────────────
wc_f = test_f[(test_f["is_world_cup"] == 1) & (test_f["is_knockout"] == 0)].copy()
wc_w = test_w[(test_w["is_world_cup"] == 1) & (test_w["is_knockout"] == 0)].copy()
y_wc = wc_f["result"].values

if len(y_wc):
    pf_wc = model_full.predict_proba(wc_f)
    pw_wc = model_win.predict_proba(wc_w)
    print("=" * 65)
    print("  WC 2026 GROUP STAGE (test set)")
    print("=" * 65)
    rows_wc = pd.DataFrame([
        metrics("full",  pf_wc, y_wc, theta_f),
        metrics("1990+", pw_wc, y_wc, theta_w),
    ])
    print(rows_wc.to_string(index=False))
    print()

# ── coefficient comparison ────────────────────────────────────────────────────
print("=" * 65)
print("  COEFFICIENTS")
print("=" * 65)
cf = model_full.coef_summary()
cw = model_win.coef_summary()
shared = ["target", "intercept", "elo_diff_scaled", "neutral",
          "h2h_home_goals_mu", "h2h_away_goals_mu", "is_world_cup", "is_knockout"]
cf = cf[[c for c in shared if c in cf.columns]]
cw = cw[[c for c in shared if c in cw.columns]]
print("  full:")
print(cf.to_string(index=False, float_format="{:.4f}".format))
print()
print("  1990+:")
print(cw.to_string(index=False, float_format="{:.4f}".format))
print()
print(f"  rho  full={model_full.rho_:.4f}   1990+={model_win.rho_:.4f}")
print()

# ── verdict ───────────────────────────────────────────────────────────────────
print("=" * 65)
print("  VERDICT")
print("=" * 65)
ll_win  = log_loss(y_test, pw_test)
ll_full = log_loss(y_test, pf_test)
f1_win_t  = f1_score(y_test, apply_threshold(pw_test, theta_w),
                     labels=CLASSES, average="macro", zero_division=0)
f1_full_t = f1_score(y_test, apply_threshold(pf_test, theta_f),
                     labels=CLASSES, average="macro", zero_division=0)
print(f"  log-loss (test):  full={ll_full:.4f}  1990+={ll_win:.4f}  "
      f"delta={ll_win-ll_full:+.4f}  {'1990+ wins' if ll_win < ll_full else 'full wins'}")
print(f"  F1-macro (test):  full={f1_full_t:.4f}  1990+={f1_win_t:.4f}  "
      f"delta={f1_win_t-f1_full_t:+.4f}  {'1990+ wins' if f1_win_t > f1_full_t else 'full wins'}")
print()
