"""
Sweep HOME_ADVANTAGE (Elo points) from 20 to 160 in steps of 10.
For each value: recompute Elo → fit v2 Poisson on train → log-loss on val.

The reliability diagram showed P(H) is systematically over-predicted by
~5-7%, suggesting HOME_ADVANTAGE=100 is too aggressive.

Split: train < 2022 | val = 2022 (same as calibrate_threshold.py)

Run from the project root:
    py recalibrate_ha.py
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.data.load import load_results
from src.features.elo import compute_elo
from src.features.h2h import compute_h2h
from src.features.poisson import PoissonDC

TRAIN_END = "2022-01-01"
VAL_END   = "2023-01-01"
CLASSES   = ["H", "D", "A"]


def label_result(hs: int, as_: int) -> str:
    if hs > as_: return "H"
    if hs < as_: return "A"
    return "D"


def log_loss(y_true, y_proba):
    idx = np.array([CLASSES.index(y) for y in y_true])
    p = y_proba[np.arange(len(y_true)), idx]
    return float(-np.mean(np.log(np.maximum(p, 1e-15))))


# ── load raw data once ────────────────────────────────────────────────
df_raw = load_results()
df_raw["result"] = df_raw.apply(
    lambda r: label_result(r.home_score, r.away_score), axis=1
)

# ── sweep ─────────────────────────────────────────────────────────────
ha_values = list(range(20, 161, 10))
rows = []

print(f"Sweeping HOME_ADVANTAGE in {ha_values}\n")

for ha in ha_values:
    # Recompute Elo with this home advantage
    df_elo, _ = compute_elo(df_raw, home_advantage=float(ha))

    # H2H priors from training data
    train_mask = df_elo["date"] < TRAIN_END
    g_home = df_elo.loc[train_mask, "home_score"].mean()
    g_away = df_elo.loc[train_mask, "away_score"].mean()

    df_feat = compute_h2h(df_elo, g_home, g_away, k=5.0)

    train = df_feat[train_mask].copy()
    val   = df_feat[(df_feat["date"] >= TRAIN_END) & (df_feat["date"] < VAL_END)].copy()

    model = PoissonDC(
        extra_home=["h2h_home_goals_mu"],
        extra_away=["h2h_away_goals_mu"],
    ).fit(train)

    val_proba = model.predict_proba(val)
    val_y     = val["result"].values

    ll = log_loss(val_y, val_proba)

    # Draw rate without threshold (natural P(D))
    draw_pred = val_proba[:, 1].mean()
    draw_act  = (val_y == "D").mean()

    rows.append({
        "home_advantage": ha,
        "log_loss": round(ll, 5),
        "draw_rate_pred": round(draw_pred, 4),
        "draw_rate_actual": round(draw_act, 4),
        "rho": round(model.rho_, 4),
    })
    print(f"  HA={ha:>4}  log_loss={ll:.5f}  draw_pred={draw_pred:.3f}  rho={model.rho_:.4f}")

results = pd.DataFrame(rows)
best = results.loc[results["log_loss"].idxmin()]

print()
print("-- Sweep results -----------------------------------")
print(results.to_string(index=False))
print()
print(f"Best HOME_ADVANTAGE = {int(best['home_advantage'])}  "
      f"(log_loss = {best['log_loss']:.5f})")
print(f"Current value (100) log_loss = "
      f"{results.loc[results['home_advantage']==100, 'log_loss'].values[0]:.5f}")
