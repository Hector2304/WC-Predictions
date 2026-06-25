"""
Train v4 model and save to models/.

Changes vs v3:
  - is_major_tourn (single binary) replaced by 6 per-tournament dummies:
    is_world_cup, is_euros, is_copa_am, is_afcon, is_asian_cup, is_gold_cup
  - tournament.py keyword matching fixed (Viva WC, CONIFA, Central European
    Cup, West African Cup no longer tagged as major)
  - Everything else identical to v3 (HA=20, H2H_K=5, train cutoff 2023-01-01)

Saved artifacts:
  models/poisson_dc_v4.joblib
  models/v4_config.json

Run from the project root:
    py save_v4.py
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
from src.features.tournament import add_tournament_features, TOURNAMENT_DUMMIES
from src.features.poisson import PoissonDC

HOME_ADVANTAGE = 20.0
TRAIN_CUTOFF   = "2023-01-01"
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
df = compute_form(df, global_avg, window=FORM_WINDOW, k=FORM_K)
df = add_tournament_features(df)

train = df[train_mask].copy()
val   = df[val_mask].copy()

# ── fit ───────────────────────────────────────────────────────────────────────
# Each dummy gets its own coefficient for home and away goal rates.
model = PoissonDC(
    extra_home=["h2h_home_goals_mu"] + TOURNAMENT_DUMMIES,
    extra_away=["h2h_away_goals_mu"] + TOURNAMENT_DUMMIES,
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
joblib.dump(model, "models/poisson_dc_v4.joblib")

config = {
    "version": "v4",
    "train_cutoff": TRAIN_CUTOFF,
    "home_advantage": HOME_ADVANTAGE,
    "features": {
        "home": ["elo_diff_scaled", "neutral", "h2h_home_goals_mu"] + TOURNAMENT_DUMMIES,
        "away": ["elo_diff_scaled", "neutral", "h2h_away_goals_mu"] + TOURNAMENT_DUMMIES,
    },
    "tournament_dummies": TOURNAMENT_DUMMIES,
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
    "theta_D": theta_d,
    "rho": round(model.rho_, 6),
}
with open("models/v4_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Saved:")
print("  models/poisson_dc_v4.joblib")
print("  models/v4_config.json")
print()
print(f"  home_advantage = {HOME_ADVANTAGE}")
print(f"  rho            = {model.rho_:.4f}")
print(f"  theta_D        = {theta_d:.2f}  (val F1_macro = {best_f1:.4f})")
print()
print("v4 coefficients:")
print(model.coef_summary().to_string(index=False, float_format="{:.4f}".format))
