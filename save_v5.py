"""
Train v5 model and save to models/.

Changes vs v4:
  - Features simplified to WC-only: is_world_cup + is_knockout
  - Drops Euros/AFCON/Copa Am dummies — product focus is World Cup only
  - is_major_tourn (backward compat alias) now equals is_world_cup
  - is_knockout derived from wc_phases.py (dataset-derived dates, 1986-2022)

Full 49k match dataset still used for Elo + H2H training.
Poisson model trained on all matches but with WC-specific tournament features.

Saved artifacts:
  models/poisson_dc_v5.joblib
  models/v5_config.json

Run from the project root:
    py save_v5.py
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
from src.features.tournament import add_tournament_features, WC_FEATURES
from src.features.poisson import PoissonDC

HOME_ADVANTAGE = 20.0
TRAIN_CUTOFF   = "2026-06-10"
VAL_START      = "2022-01-01"
CLASSES        = ["H", "D", "A"]
H2H_K          = 5.0
FORM_WINDOW    = 5
FORM_K         = 3.0


def label_result(hs: int, as_: int) -> str:
    if hs > as_: return "H"
    if hs < as_: return "A"
    return "D"


def apply_threshold(proba: np.ndarray, theta_d: float) -> np.ndarray:
    p_h, p_d, p_a = proba[:, 0], proba[:, 1], proba[:, 2]
    return np.where(p_d > theta_d, "D", np.where(p_h >= p_a, "H", "A"))


# ── build features ────────────────────────────────────────────────────────────
df_raw = load_results()
df, _  = compute_elo(df_raw, home_advantage=HOME_ADVANTAGE)
df["result"] = df.apply(lambda r: label_result(r.home_score, r.away_score), axis=1)

train_mask = df["date"] < TRAIN_CUTOFF
val_mask   = (df["date"] >= VAL_START) & (df["date"] < TRAIN_CUTOFF)

train_raw       = df[train_mask]
global_home_avg = train_raw["home_score"].mean()
global_away_avg = train_raw["away_score"].mean()
global_avg      = (global_home_avg + global_away_avg) / 2

df = compute_h2h(df, global_home_avg, global_away_avg, k=H2H_K)
df = add_tournament_features(df)

train = df[train_mask].copy()
val   = df[val_mask].copy()

print(f"Training on {train_mask.sum():,} matches")
wc_train    = train["is_world_cup"].sum()
ko_train    = train["is_knockout"].sum()
group_train = wc_train - ko_train
print(f"  WC matches in train: {int(wc_train)} ({int(group_train)} group, {int(ko_train)} knockout)")
print()

# ── fit ───────────────────────────────────────────────────────────────────────
# WC_FEATURES = ["is_world_cup", "is_knockout"]
model = PoissonDC(
    extra_home=["h2h_home_goals_mu"] + WC_FEATURES,
    extra_away=["h2h_away_goals_mu"] + WC_FEATURES,
)
model.fit(train)

# ── calibrate theta_D on val ──────────────────────────────────────────────────
val_proba = model.predict_proba(val)
val_y     = val["result"].values

best_f1, theta_d = -1.0, 0.25
for t in np.round(np.arange(0.10, 0.46, 0.01), 2):
    pred = apply_threshold(val_proba, t)
    f1   = f1_score(val_y, pred, labels=CLASSES, average="macro", zero_division=0)
    if f1 > best_f1:
        best_f1, theta_d = f1, float(t)

# ── persist ───────────────────────────────────────────────────────────────────
joblib.dump(model, "models/poisson_dc_v5.joblib")

config = {
    "version": "v5",
    "train_cutoff": TRAIN_CUTOFF,
    "home_advantage": HOME_ADVANTAGE,
    "features": {
        "home": ["elo_diff_scaled", "neutral", "h2h_home_goals_mu"] + WC_FEATURES,
        "away": ["elo_diff_scaled", "neutral", "h2h_away_goals_mu"] + WC_FEATURES,
    },
    "wc_features": WC_FEATURES,
    "priors": {
        "global_home_avg": round(global_home_avg, 4),
        "global_away_avg": round(global_away_avg, 4),
        "global_avg":      round(global_avg, 4),
    },
    "hyperparams": {
        "h2h_k":       H2H_K,
        "form_window": FORM_WINDOW,
        "form_k":      FORM_K,
    },
    "theta_D":          theta_d,
    "theta_D_knockout": 0.28,
    "rho": round(model.rho_, 6),
}
with open("models/v5_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Saved:")
print("  models/poisson_dc_v5.joblib")
print("  models/v5_config.json")
print()
print(f"  home_advantage = {HOME_ADVANTAGE}")
print(f"  rho            = {model.rho_:.4f}")
print(f"  theta_D        = {theta_d:.2f}  (val F1_macro = {best_f1:.4f})")
print()
print("v5 coefficients:")
print(model.coef_summary().to_string(index=False, float_format="{:.4f}".format))
