"""
Side-by-side comparison of v2 and v3 on the test set (2023+).

Focus areas matching the diagnoses from backtest.py:
  - Overall metrics
  - Euros / AFCON / Copa Am gap  (the main motivation for v3)
  - Reliability diagram: P(H) and P(D) calibration
  - Draw rate by tournament

Run from the project root:
    py compare_v2_v3.py
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
from src.features.tournament import add_tournament_features

TEST_START = "2023-01-01"
CLASSES    = ["H", "D", "A"]


# ── helpers ───────────────────────────────────────────────────────────

def label_result(hs: int, as_: int) -> str:
    if hs > as_: return "H"
    if hs < as_: return "A"
    return "D"


def apply_threshold(proba: np.ndarray, theta_d: float) -> np.ndarray:
    p_h, p_d, p_a = proba[:, 0], proba[:, 1], proba[:, 2]
    return np.where(p_d > theta_d, "D", np.where(p_h >= p_a, "H", "A"))


def log_loss(y_true, y_proba):
    idx = np.array([CLASSES.index(y) for y in y_true])
    p = y_proba[np.arange(len(y_true)), idx]
    return float(-np.mean(np.log(np.maximum(p, 1e-15))))


def tournament_cat(tournament: str) -> str:
    t = tournament.lower()
    if "world cup" in t and "qual" not in t:     return "World Cup"
    if "qual" in t or "qualifier" in t:          return "WC Qualifier"
    if "euro" in t and "qual" not in t:          return "UEFA Euro"
    if "copa am" in t and "qual" not in t:       return "Copa America"
    if "africa cup" in t or "african cup" in t:  return "AFCON"
    if "asian cup" in t:                         return "Asian Cup"
    if "gold cup" in t:                          return "Gold Cup"
    if "nations league" in t:                    return "Nations League"
    if "friendly" in t:                          return "Friendly"
    return "Other"


def build_features(df_raw, cfg):
    ha = cfg.get("home_advantage", 100.0)
    df, _ = compute_elo(df_raw, home_advantage=ha)
    df["result"] = df.apply(lambda r: label_result(r.home_score, r.away_score), axis=1)
    p = cfg["priors"]
    hp = cfg["hyperparams"]
    df = compute_h2h(df, p["global_home_avg"], p["global_away_avg"], k=hp["h2h_k"])
    df = compute_form(df, p["global_avg"], window=hp["form_window"], k=hp["form_k"])
    df = add_tournament_features(df)
    return df


def eval_model(model, cfg, df_full):
    test = df_full[df_full["date"] >= TEST_START].copy()
    test["tournament_cat"] = test["tournament"].apply(tournament_cat)
    proba = model.predict_proba(test)
    pred  = apply_threshold(proba, cfg["theta_D"])
    y     = test["result"].values
    return test, proba, pred, y


# ── load models ───────────────────────────────────────────────────────
m2 = joblib.load("models/poisson_dc_v2.joblib")
m3 = joblib.load("models/poisson_dc_v3.joblib")
with open("models/v2_config.json") as f: cfg2 = json.load(f)
with open("models/v3_config.json") as f: cfg3 = json.load(f)

df_raw = load_results()

# Build features for each model (different HOME_ADVANTAGE)
df2 = build_features(df_raw, cfg2)
df3 = build_features(df_raw, cfg3)

test2, proba2, pred2, y2 = eval_model(m2, cfg2, df2)
test3, proba3, pred3, y3 = eval_model(m3, cfg3, df3)

print(f"v2: HA={cfg2.get('home_advantage',100)}  theta_D={cfg2['theta_D']}  rho={cfg2['rho']}")
print(f"v3: HA={cfg3['home_advantage']}  theta_D={cfg3['theta_D']}  rho={cfg3['rho']}\n")

# ── 1. Overall ────────────────────────────────────────────────────────
def row(label, y, pred, proba):
    return {
        "model": label,
        "accuracy":  round(accuracy_score(y, pred), 3),
        "f1_macro":  round(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0), 3),
        "f1_draw":   round(f1_score(y, pred, labels=["D"], average="macro", zero_division=0), 3),
        "log_loss":  round(log_loss(y, proba), 4),
        "draw_pred": round((pred == "D").mean(), 3),
        "draw_act":  round((y == "D").mean(), 3),
    }

overall = pd.DataFrame([row("v2", y2, pred2, proba2), row("v3", y3, pred3, proba3)])
print("-- Overall (test >= 2023) ---------------------------")
print(overall.to_string(index=False))
print()

# ── 2. Tournament breakdown ───────────────────────────────────────────
target_tourneys = ["UEFA Euro", "AFCON", "Copa America", "World Cup",
                   "WC Qualifier", "Nations League", "Friendly"]

tourn_rows = []
for tc in target_tourneys:
    mask2 = test2["tournament_cat"] == tc
    mask3 = test3["tournament_cat"] == tc
    if mask2.sum() < 5:
        continue
    y_t   = y2[mask2]
    pr2   = proba2[mask2]
    pr3   = proba3[mask3]
    pd2   = apply_threshold(pr2, cfg2["theta_D"])
    pd3   = apply_threshold(pr3, cfg3["theta_D"])
    tourn_rows.append({
        "tournament": tc,
        "n": mask2.sum(),
        "ll_v2":  round(log_loss(y_t, pr2), 4),
        "ll_v3":  round(log_loss(y_t, pr3), 4),
        "ll_delta": round(log_loss(y_t, pr2) - log_loss(y_t, pr3), 4),
        "draw_act":  round((y_t == "D").mean(), 3),
        "draw_v2":   round((pd2 == "D").mean(), 3),
        "draw_v3":   round((pd3 == "D").mean(), 3),
    })

tourn_df = pd.DataFrame(tourn_rows)
print("-- Tournament breakdown (ll_delta = v2 - v3, positive = v3 wins) --")
print(tourn_df.to_string(index=False))
print()

# ── 3. P(H) reliability: v2 vs v3 ────────────────────────────────────
print("-- P(H) reliability: delta = actual - predicted ----")
print(f"  {'bin':<10}  {'delta_v2':>10}  {'delta_v3':>10}  {'improvement':>12}")
for lo in np.arange(0.2, 0.9, 0.1):
    hi = lo + 0.1
    m2_bin = (proba2[:, 0] >= lo) & (proba2[:, 0] < hi)
    m3_bin = (proba3[:, 0] >= lo) & (proba3[:, 0] < hi)
    if m2_bin.sum() < 10:
        continue
    d2 = (y2[m2_bin] == "H").mean() - proba2[m2_bin, 0].mean()
    d3 = (y3[m3_bin] == "H").mean() - proba3[m3_bin, 0].mean()
    print(f"  {lo:.0%}-{hi:.0%}       {d2:>+.3f}       {d3:>+.3f}       {d3-d2:>+.3f}")
print()

# ── 4. P(D) reliability: v2 vs v3 ────────────────────────────────────
print("-- P(D) reliability: delta = actual - predicted ----")
print(f"  {'bin':<10}  {'delta_v2':>10}  {'delta_v3':>10}  {'improvement':>12}")
for lo in np.arange(0.10, 0.35, 0.05):
    hi = lo + 0.05
    m2_bin = (proba2[:, 1] >= lo) & (proba2[:, 1] < hi)
    m3_bin = (proba3[:, 1] >= lo) & (proba3[:, 1] < hi)
    if m2_bin.sum() < 10:
        continue
    d2 = (y2[m2_bin] == "D").mean() - proba2[m2_bin, 1].mean()
    d3 = (y3[m3_bin] == "D").mean() - proba3[m3_bin, 1].mean()
    print(f"  {lo:.0%}-{hi:.0%}     {d2:>+.3f}       {d3:>+.3f}       {d3-d2:>+.3f}")
