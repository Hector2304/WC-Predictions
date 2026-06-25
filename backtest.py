"""
Comprehensive back-test of v2 model on 2023-2024 data.

Loads the serialised model from models/ and evaluates four angles:
  1. Overall performance (sanity check)
  2. Quarterly breakdown  — is performance stable over time?
  3. Major tournament breakdown — does it hold in high-stakes matches?
  4. Reliability diagram — are probabilities actually calibrated?
  5. Top missed calls — biggest upsets the model got wrong

Run from the project root:
    py backtest.py
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


def log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    idx = np.array([CLASSES.index(y) for y in y_true])
    p = y_proba[np.arange(len(y_true)), idx]
    return float(-np.mean(np.log(np.maximum(p, 1e-15))))


def metrics(y_true, pred, proba) -> dict:
    return {
        "n":         len(y_true),
        "accuracy":  round(accuracy_score(y_true, pred), 3),
        "f1_macro":  round(f1_score(y_true, pred, labels=CLASSES, average="macro", zero_division=0), 3),
        "f1_draw":   round(f1_score(y_true, pred, labels=["D"], average="macro", zero_division=0), 3),
        "log_loss":  round(log_loss(y_true, proba), 4),
        "draw_pred": round((pred == "D").mean(), 3),
        "draw_act":  round((y_true == "D").mean(), 3),
    }


def tournament_cat(tournament: str) -> str:
    t = tournament.lower()
    if "world cup" in t and "qual" not in t:       return "World Cup"
    if "qual" in t or "qualifier" in t:            return "WC Qualifier"
    if "euro" in t and "qual" not in t:            return "UEFA Euro"
    if "copa am" in t and "qual" not in t:         return "Copa America"
    if "africa cup" in t or "african cup" in t:    return "AFCON"
    if "asian cup" in t:                           return "Asian Cup"
    if "gold cup" in t:                            return "Gold Cup"
    if "nations league" in t:                      return "Nations League"
    if "friendly" in t:                            return "Friendly"
    return "Other"


# ── load model + config ───────────────────────────────────────────────
model = joblib.load("models/poisson_dc_v2.joblib")
with open("models/v2_config.json") as f:
    cfg = json.load(f)

theta_d       = cfg["theta_D"]
global_home   = cfg["priors"]["global_home_avg"]
global_away   = cfg["priors"]["global_away_avg"]
global_avg    = cfg["priors"]["global_avg"]
h2h_k         = cfg["hyperparams"]["h2h_k"]
form_window   = cfg["hyperparams"]["form_window"]
form_k        = cfg["hyperparams"]["form_k"]

print(f"Model: {cfg['version']}  |  rho={cfg['rho']}  |  theta_D={theta_d}")
print(f"Train cutoff: {cfg['train_cutoff']}\n")

# ── rebuild features ──────────────────────────────────────────────────
df_raw = load_results()
df, _ = compute_elo(df_raw)
df["result"] = df.apply(lambda r: label_result(r.home_score, r.away_score), axis=1)

df_h2h  = compute_h2h(df, global_home, global_away, k=h2h_k)
df_full = compute_form(df_h2h, global_avg, window=form_window, k=form_k)

test = df_full[df_full["date"] >= TEST_START].copy()
test["tournament_cat"] = test["tournament"].apply(tournament_cat)
test["quarter"] = test["date"].dt.to_period("Q").astype(str)

test_proba = model.predict_proba(test)
test_pred  = apply_threshold(test_proba, theta_d)
test_y     = test["result"].values

test["p_H"]  = test_proba[:, 0]
test["p_D"]  = test_proba[:, 1]
test["p_A"]  = test_proba[:, 2]
test["pred"] = test_pred

print(f"Test set: {len(test):,} matches ({TEST_START} onwards)\n")

# ── 1. Overall ────────────────────────────────────────────────────────
m = metrics(test_y, test_pred, test_proba)
print("-- Overall performance -----------------------------")
for k_name, v in m.items():
    print(f"  {k_name:<12} {v}")
print()

# ── 2. Quarterly breakdown ────────────────────────────────────────────
print("-- Quarterly breakdown -----------------------------")
qtrs = []
for q, grp in test.groupby("quarter"):
    y = grp["result"].values
    proba = np.column_stack([grp["p_H"], grp["p_D"], grp["p_A"]])
    pred  = apply_threshold(proba, theta_d)
    row = {"quarter": q}
    row.update(metrics(y, pred, proba))
    qtrs.append(row)

qtr_df = pd.DataFrame(qtrs)
print(qtr_df[["quarter", "n", "accuracy", "f1_macro", "f1_draw", "log_loss",
              "draw_pred", "draw_act"]].to_string(index=False))
print()

# ── 3. Tournament breakdown ───────────────────────────────────────────
print("-- Tournament breakdown ----------------------------")
tourn_rows = []
for tc, grp in test.groupby("tournament_cat"):
    y = grp["result"].values
    proba = np.column_stack([grp["p_H"], grp["p_D"], grp["p_A"]])
    pred  = apply_threshold(proba, theta_d)
    row = {"tournament": tc}
    row.update(metrics(y, pred, proba))
    tourn_rows.append(row)

tourn_df = pd.DataFrame(tourn_rows).sort_values("n", ascending=False)
print(tourn_df[["tournament", "n", "accuracy", "f1_macro", "log_loss",
                "draw_pred", "draw_act"]].to_string(index=False))
print()

# ── 4. Reliability diagram ────────────────────────────────────────────
print("-- Reliability diagram (P(H) calibration) ---------")
print(f"  {'bin':<12} {'n':>6} {'pred_p_h':>10} {'actual_h':>10} {'delta':>8}")
for lo in np.arange(0, 1.0, 0.1):
    hi = lo + 0.1
    mask = (test["p_H"] >= lo) & (test["p_H"] < hi)
    if mask.sum() < 5:
        continue
    pred_p = test.loc[mask, "p_H"].mean()
    actual = (test.loc[mask, "result"] == "H").mean()
    delta  = actual - pred_p
    print(f"  {lo:.0%}-{hi:.0%}      {mask.sum():>6}     {pred_p:.3f}     {actual:.3f}   {delta:+.3f}")
print()

print("-- Reliability diagram (P(D) calibration) ---------")
print(f"  {'bin':<12} {'n':>6} {'pred_p_d':>10} {'actual_d':>10} {'delta':>8}")
for lo in np.arange(0, 0.5, 0.05):
    hi = lo + 0.05
    mask = (test["p_D"] >= lo) & (test["p_D"] < hi)
    if mask.sum() < 5:
        continue
    pred_p = test.loc[mask, "p_D"].mean()
    actual = (test.loc[mask, "result"] == "D").mean()
    delta  = actual - pred_p
    print(f"  {lo:.0%}-{hi:.0%}    {mask.sum():>6}     {pred_p:.3f}     {actual:.3f}   {delta:+.3f}")
print()

# ── 5. Top missed calls ───────────────────────────────────────────────
# Matches where model was most confident AND wrong (highest |log prob of actual|)
test["log_p_actual"] = test.apply(
    lambda r: np.log(max({"H": r.p_H, "D": r.p_D, "A": r.p_A}[r.result], 1e-15)), axis=1
)
test["correct"] = test["pred"] == test["result"]

worst = (
    test[~test["correct"]]
    .nsmallest(12, "log_p_actual")
    [["date", "home_team", "away_team", "home_score", "away_score",
      "tournament_cat", "pred", "result", "p_H", "p_D", "p_A"]]
)
print("-- Top 12 missed calls (model most confident, still wrong) --")
print(worst.to_string(index=False, float_format="{:.3f}".format))
