"""
Train v2 model on all data before 2023 and persist to models/.

Saved artifacts:
  models/poisson_dc_v2.joblib  — fitted PoissonDC (sklearn + rho)
  models/v2_config.json        — priors, theta_D, feature spec, metadata

To reproduce predictions on any future data:
  1. Load raw data, run compute_elo + compute_h2h using the saved priors.
  2. Load the model from joblib.
  3. Call model.predict_proba(df).

Run from the project root:
    py save_pipeline.py
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
from src.features.poisson import PoissonDC

TRAIN_CUTOFF = "2023-01-01"
VAL_START    = "2022-01-01"   # val set for theta_D calibration
CLASSES      = ["H", "D", "A"]
H2H_K        = 5.0
FORM_WINDOW  = 5
FORM_K       = 3.0


def label_result(hs: int, as_: int) -> str:
    if hs > as_: return "H"
    if hs < as_: return "A"
    return "D"


def apply_threshold(proba: np.ndarray, theta_d: float) -> np.ndarray:
    p_h, p_d, p_a = proba[:, 0], proba[:, 1], proba[:, 2]
    return np.where(p_d > theta_d, "D", np.where(p_h >= p_a, "H", "A"))


# ── build full feature set ────────────────────────────────────────────
df_raw = load_results()
df, _ = compute_elo(df_raw)
df["result"] = df.apply(lambda r: label_result(r.home_score, r.away_score), axis=1)

train_mask = df["date"] < TRAIN_CUTOFF
val_mask   = (df["date"] >= VAL_START) & (df["date"] < TRAIN_CUTOFF)

# Priors from full training data (pre-2023)
train_raw = df[train_mask]
global_home_avg = train_raw["home_score"].mean()
global_away_avg = train_raw["away_score"].mean()
global_avg      = (global_home_avg + global_away_avg) / 2

df_h2h  = compute_h2h(df, global_home_avg, global_away_avg, k=H2H_K)
df_full = compute_form(df_h2h, global_avg, window=FORM_WINDOW, k=FORM_K)

train = df_full[train_mask].copy()
val   = df_full[val_mask].copy()

# ── fit v2 ────────────────────────────────────────────────────────────
model = PoissonDC(
    extra_home=["h2h_home_goals_mu"],
    extra_away=["h2h_away_goals_mu"],
)
model.fit(train)

# ── calibrate theta_D on val ─────────────────────────────────────────
val_proba = model.predict_proba(val)
val_y     = val["result"].values

best_f1, theta_d = -1.0, 0.25
for t in np.round(np.arange(0.10, 0.46, 0.01), 2):
    pred = apply_threshold(val_proba, t)
    f1 = f1_score(val_y, pred, labels=CLASSES, average="macro", zero_division=0)
    if f1 > best_f1:
        best_f1, theta_d = f1, float(t)

# ── persist ───────────────────────────────────────────────────────────
joblib.dump(model, "models/poisson_dc_v2.joblib")

config = {
    "version": "v2",
    "train_cutoff": TRAIN_CUTOFF,
    "features": {
        "home": ["elo_diff_scaled", "neutral", "h2h_home_goals_mu"],
        "away": ["elo_diff_scaled", "neutral", "h2h_away_goals_mu"],
    },
    "priors": {
        "global_home_avg": round(global_home_avg, 4),
        "global_away_avg": round(global_away_avg, 4),
        "global_avg":      round(global_avg, 4),
    },
    "hyperparams": {
        "h2h_k":      H2H_K,
        "form_window": FORM_WINDOW,
        "form_k":      FORM_K,
    },
    "theta_D": theta_d,
    "rho":     round(model.rho_, 6),
}
with open("models/v2_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Saved:")
print("  models/poisson_dc_v2.joblib")
print("  models/v2_config.json")
print()
print(f"  rho     = {model.rho_:.4f}")
print(f"  theta_D = {theta_d:.2f}  (val F1_macro = {best_f1:.4f})")
print()
print("v2 coefficients:")
print(model.coef_summary().to_string(index=False, float_format="{:.4f}".format))
