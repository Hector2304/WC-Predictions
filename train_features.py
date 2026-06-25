"""
Feature ablation study: how much does each feature group add?

Models evaluated:
  v1  baseline : elo_diff + neutral
  v2  + h2h    : v1 + h2h_home_goals_mu + h2h_away_goals_mu
  v3  + form   : v2 + form_home_attack + form_home_defense
                    + form_away_attack + form_away_defense

Split: train < 2022 | val = 2022 (calibrates theta_D) | test >= 2023

Metrics: log-loss (no threshold) and F1 macro (with calibrated theta_D).

Run from the project root:
    py train_features.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.data.load import load_results
from src.features.elo import compute_elo
from src.features.h2h import compute_h2h
from src.features.form import compute_form
from src.features.poisson import PoissonDC

TRAIN_END = "2022-01-01"
VAL_END   = "2023-01-01"
CLASSES   = ["H", "D", "A"]


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


def best_theta(val_proba: np.ndarray, val_y: np.ndarray) -> float:
    """Maximise F1 macro over theta_D in [0.10, 0.45]."""
    best_f1, best_t = -1.0, 0.25
    for t in np.round(np.arange(0.10, 0.46, 0.01), 2):
        pred = apply_threshold(val_proba, t)
        f1 = f1_score(val_y, pred, labels=CLASSES, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def evaluate(name: str, model: PoissonDC, val_df, test_df, val_y, test_y) -> dict:
    val_proba  = model.predict_proba(val_df)
    test_proba = model.predict_proba(test_df)

    theta = best_theta(val_proba, val_y)
    test_pred = apply_threshold(test_proba, theta)

    return {
        "model": name,
        "rho": round(model.rho_, 4),
        "theta_D": theta,
        "log_loss": round(log_loss(test_y, test_proba), 4),
        "accuracy": round(accuracy_score(test_y, test_pred), 3),
        "f1_macro": round(f1_score(test_y, test_pred, labels=CLASSES, average="macro", zero_division=0), 3),
        "f1_draw":  round(f1_score(test_y, test_pred, labels=["D"], average="macro", zero_division=0), 3),
        "draw_pred_rate": round((test_pred == "D").mean(), 3),
    }


# ── load & Elo ────────────────────────────────────────────────────────
df_raw = load_results()
df, _ = compute_elo(df_raw)
df["result"] = df.apply(lambda r: label_result(r.home_score, r.away_score), axis=1)

train_mask = df["date"] < TRAIN_END
val_mask   = (df["date"] >= TRAIN_END) & (df["date"] < VAL_END)
test_mask  = df["date"] >= VAL_END

train_base = df[train_mask]
print(f"Train: {train_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}\n")

# Global priors computed from training set (no leakage)
global_home_avg = train_base["home_score"].mean()
global_away_avg = train_base["away_score"].mean()
global_avg      = (global_home_avg + global_away_avg) / 2

print(f"Priors  home_avg={global_home_avg:.3f}  away_avg={global_away_avg:.3f}  "
      f"combined={global_avg:.3f}\n")

# ── build feature layers ──────────────────────────────────────────────
# H2H and form run on the full df (chronological logic handles leakage).
df_h2h  = compute_h2h(df, global_home_avg, global_away_avg, k=5.0)
df_full = compute_form(df_h2h, global_avg, window=5, k=3.0)

train = df_full[train_mask].copy()
val   = df_full[val_mask].copy()
test  = df_full[test_mask].copy()

val_y  = val["result"].values
test_y = test["result"].values

# ── v1: baseline ──────────────────────────────────────────────────────
m1 = PoissonDC().fit(train)
r1 = evaluate("v1_baseline", m1, val, test, val_y, test_y)

# ── v2: + h2h ─────────────────────────────────────────────────────────
m2 = PoissonDC(
    extra_home=["h2h_home_goals_mu"],
    extra_away=["h2h_away_goals_mu"],
).fit(train)
r2 = evaluate("v2_+h2h", m2, val, test, val_y, test_y)

# ── v3: + form ────────────────────────────────────────────────────────
m3 = PoissonDC(
    extra_home=["h2h_home_goals_mu", "form_home_attack", "form_away_defense"],
    extra_away=["h2h_away_goals_mu", "form_away_attack", "form_home_defense"],
).fit(train)
r3 = evaluate("v3_+form", m3, val, test, val_y, test_y)

# ── results table ─────────────────────────────────────────────────────
results = pd.DataFrame([r1, r2, r3])
print("-- Ablation study (test >= 2023) -------------------")
print(results.to_string(index=False))
print()

# ── delta from baseline ───────────────────────────────────────────────
for r, label in [(r2, "v2 vs v1"), (r3, "v3 vs v1")]:
    ll_delta = r1["log_loss"] - r["log_loss"]
    f1_delta = r["f1_macro"] - r1["f1_macro"]
    print(f"{label}:  log_loss delta={ll_delta:+.4f}  f1_macro delta={f1_delta:+.3f}")
print()

# ── v3 coefficients ───────────────────────────────────────────────────
print("-- v3 coefficients (best model) --------------------")
print(m3.coef_summary().to_string(index=False, float_format="{:.4f}".format))
print()

# ── h2h coverage on test ──────────────────────────────────────────────
print("-- H2H coverage on test set ------------------------")
h2h_dist = test["h2h_n"].describe()
print(h2h_dist.to_string())
print(f"  Matches with 0 H2H history: {(test['h2h_n'] == 0).mean():.1%}")
print(f"  Matches with >=5 H2H:       {(test['h2h_n'] >= 5).mean():.1%}")
