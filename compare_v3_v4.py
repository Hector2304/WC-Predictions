"""
Side-by-side comparison of v3 vs v4 on the test set (2023+).

Key question: do per-tournament dummies close the draw calibration gap
in Euros / AFCON / WC more than the single is_major_tourn flag?
If yes, the is_knockout table is not needed.

Run from the project root:
    py compare_v3_v4.py
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


# ── helpers ───────────────────────────────────────────────────────────────────

def label_result(hs: int, as_: int) -> str:
    if hs > as_: return "H"
    if hs < as_: return "A"
    return "D"


def apply_threshold(proba: np.ndarray, theta_d: float) -> np.ndarray:
    p_h, p_d, p_a = proba[:, 0], proba[:, 1], proba[:, 2]
    return np.where(p_d > theta_d, "D", np.where(p_h >= p_a, "H", "A"))


def log_loss(y_true, y_proba):
    idx = np.array([CLASSES.index(y) for y in y_true])
    p   = y_proba[np.arange(len(y_true)), idx]
    return float(-np.mean(np.log(np.maximum(p, 1e-15))))


def tournament_cat(tournament: str) -> str:
    t = tournament.lower()
    if "fifa world cup" in t and "qual" not in t:      return "World Cup"
    if "qual" in t or "qualifier" in t:                return "Qualifier"
    if "uefa euro" in t:                               return "UEFA Euro"
    if "copa am" in t:                                 return "Copa America"
    if "african cup of nations" in t:                  return "AFCON"
    if "afc asian cup" in t:                           return "Asian Cup"
    if "gold cup" in t:                                return "Gold Cup"
    if "nations league" in t:                          return "Nations League"
    if "friendly" in t:                                return "Friendly"
    return "Other"


def summary_row(label, y, pred, proba):
    return {
        "model":     label,
        "n":         len(y),
        "log_loss":  round(log_loss(y, proba), 4),
        "accuracy":  round(accuracy_score(y, pred), 3),
        "f1_macro":  round(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0), 3),
        "f1_draw":   round(f1_score(y, pred, labels=["D"], average="macro", zero_division=0), 3),
        "draw_pred": round((pred == "D").mean(), 3),
        "draw_act":  round((y == "D").mean(), 3),
    }


# ── load data + features (built once, both models share same HA) ──────────────
with open("models/v3_config.json") as f: cfg3 = json.load(f)
with open("models/v4_config.json") as f: cfg4 = json.load(f)

m3 = joblib.load("models/poisson_dc_v3.joblib")
m4 = joblib.load("models/poisson_dc_v4.joblib")

df_raw = load_results()
p3     = cfg3["priors"]
hp3    = cfg3["hyperparams"]

df, _  = compute_elo(df_raw, home_advantage=cfg3["home_advantage"])
df["result"] = df.apply(lambda r: label_result(r.home_score, r.away_score), axis=1)
df = compute_h2h(df, p3["global_home_avg"], p3["global_away_avg"], k=hp3["h2h_k"])
df = compute_form(df, p3["global_avg"], window=hp3["form_window"], k=hp3["form_k"])
df = add_tournament_features(df)

test = df[df["date"] >= TEST_START].copy()
test["cat"] = test["tournament"].apply(tournament_cat)

proba3 = m3.predict_proba(test)
proba4 = m4.predict_proba(test)
pred3  = apply_threshold(proba3, cfg3["theta_D"])
pred4  = apply_threshold(proba4, cfg4["theta_D"])
y      = test["result"].values

print(f"v3: theta_D={cfg3['theta_D']}  rho={cfg3['rho']}")
print(f"v4: theta_D={cfg4['theta_D']}  rho={cfg4['rho']}")
print()

# ── 1. Overall ────────────────────────────────────────────────────────────────
overall = pd.DataFrame([
    summary_row("v3", y, pred3, proba3),
    summary_row("v4", y, pred4, proba4),
])
print("-- Overall (test >= 2023) -----------------------------------------------")
print(overall.to_string(index=False))
print()

# ── 2. Tournament breakdown ───────────────────────────────────────────────────
target = ["World Cup", "UEFA Euro", "Copa America", "AFCON", "Asian Cup",
          "Gold Cup", "Nations League", "Qualifier", "Friendly"]

rows = []
for cat in target:
    mask = test["cat"] == cat
    n = mask.sum()
    if n < 5:
        continue
    y_t  = y[mask]
    p3_t = proba3[mask]
    p4_t = proba4[mask]
    pd3  = apply_threshold(p3_t, cfg3["theta_D"])
    pd4  = apply_threshold(p4_t, cfg4["theta_D"])
    rows.append({
        "tournament":  cat,
        "n":           n,
        "draw_act":    round((y_t == "D").mean(), 3),
        "draw_v3":     round((pd3 == "D").mean(), 3),
        "draw_v4":     round((pd4 == "D").mean(), 3),
        "ll_v3":       round(log_loss(y_t, p3_t), 4),
        "ll_v4":       round(log_loss(y_t, p4_t), 4),
        "ll_delta":    round(log_loss(y_t, p3_t) - log_loss(y_t, p4_t), 4),
    })

tdf = pd.DataFrame(rows)
print("-- Tournament breakdown (ll_delta = v3 - v4, positive = v4 wins) --------")
print(tdf.to_string(index=False))
print()

# ── 3. v4 coefficients per tournament dummy ───────────────────────────────────
print("-- v4 tournament dummy coefficients -------------------------------------")
coef = m4.coef_summary()
from src.features.tournament import TOURNAMENT_DUMMIES
dummy_cols = [c for c in coef.columns if c in TOURNAMENT_DUMMIES]
display_cols = ["target", "intercept", "elo_diff_scaled", "neutral", "h2h_home_goals_mu",
                "h2h_away_goals_mu"] + dummy_cols
display_cols = [c for c in display_cols if c in coef.columns]
print(coef[display_cols].to_string(index=False, float_format="{:.4f}".format))
print()

# ── 4. P(D) reliability: v3 vs v4 ────────────────────────────────────────────
print("-- P(D) reliability (actual - predicted, by raw-P(D) bin) ---------------")
print(f"  {'bin':<10}  {'n':>5}  {'delta_v3':>10}  {'delta_v4':>10}  {'improvement':>12}")
y_idx = np.array([CLASSES.index(yi) for yi in y])
for lo in np.arange(0.10, 0.40, 0.05):
    hi = lo + 0.05
    m3b = (proba3[:, 1] >= lo) & (proba3[:, 1] < hi)
    m4b = (proba4[:, 1] >= lo) & (proba4[:, 1] < hi)
    if m3b.sum() < 10:
        continue
    d3 = (y_idx[m3b] == 1).mean() - proba3[m3b, 1].mean()
    d4 = (y_idx[m4b] == 1).mean() - proba4[m4b, 1].mean()
    print(f"  {lo:.0%}-{hi:.0%}      {m3b.sum():>5}     {d3:>+.3f}       {d4:>+.3f}       {d4 - d3:>+.3f}")
print()

# ── 5. Draw rate gap summary ──────────────────────────────────────────────────
print("-- Draw rate gap summary ------------------------------------------------")
draw_real = (y == "D").mean()
draw_v3   = (pred3 == "D").mean()
draw_v4   = (pred4 == "D").mean()
gap_v3    = draw_v3 - draw_real
gap_v4    = draw_v4 - draw_real
print(f"  Real draw rate:  {draw_real:.1%}")
print(f"  v3 predicted:    {draw_v3:.1%}  (over-prediction: {gap_v3:+.1%})")
print(f"  v4 predicted:    {draw_v4:.1%}  (over-prediction: {gap_v4:+.1%})")
reduction = (abs(gap_v3) - abs(gap_v4)) / abs(gap_v3) if gap_v3 != 0 else 0
print(f"  Gap reduction:   {reduction:.0%}")
print()
