"""
Validate Poisson + Dixon-Coles model against the Elo baseline.

Key metrics (as specified in the design):
  1. Log-loss multiclass H/D/A  — rewards calibration, penalises overconfidence
  2. Predicted draw rate        — should approach 23.1% actual
  3. Accuracy                   — for direct comparison with Elo

Elo baseline uses expected score for P(H)/P(A) and the historical
draw rate (23.1%) as a flat prior, so it's a fair opponent.

Run from the project root:
    py validate_poisson.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.data.load import load_results
from src.features.elo import HOME_ADVANTAGE, compute_elo
from src.features.poisson import PoissonDC

TEST_START = "2023-01-01"
CLASSES = ["H", "D", "A"]
HISTORICAL_DRAW_RATE = 0.231  # from error_analysis.py


# ── helpers ───────────────────────────────────────────────────────────

def label_result(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def elo_proba_3class(elo_diff: np.ndarray, neutral: np.ndarray) -> np.ndarray:
    """
    Convert Elo difference to 3-class probabilities.
    P(D) = flat historical rate; P(H)/P(A) share remaining mass
    proportional to Elo expected score.
    """
    adj = np.where(neutral, elo_diff, elo_diff + HOME_ADVANTAGE)
    e_h = 1.0 / (1.0 + 10.0 ** (-adj / 400.0))
    e_a = 1.0 - e_h
    p_h = e_h * (1.0 - HISTORICAL_DRAW_RATE)
    p_a = e_a * (1.0 - HISTORICAL_DRAW_RATE)
    p_d = np.full(len(elo_diff), HISTORICAL_DRAW_RATE)
    return np.column_stack([p_h, p_d, p_a])


# ── load & compute Elo ───────────────────────────────────────────────
df_raw = load_results()
df, _ = compute_elo(df_raw)
df["result"] = df.apply(lambda r: label_result(r.home_score, r.away_score), axis=1)

train = df[df["date"] < TEST_START].copy()
test = df[df["date"] >= TEST_START].copy()

print(f"Train: {len(train):,}  |  Test: {len(test):,}\n")

# ── fit Poisson + DC ─────────────────────────────────────────────────
model = PoissonDC(alpha=0.0)
model.fit(train)

print(f"Estimated rho: {model.rho_:.4f}  (negative = draws more likely than pure Poisson)\n")

print("-- Model coefficients ------------------------------")
print(model.coef_summary().to_string(index=False, float_format="{:.4f}".format))
print("  Expected signs: elo_diff_per400 > 0 for home, < 0 for away")
print()

# ── predictions ──────────────────────────────────────────────────────
poisson_proba = model.predict_proba(test)           # (n, 3) → H, D, A
elo_proba = elo_proba_3class(                       # (n, 3) → H, D, A
    test["elo_diff"].values, test["neutral"].values
)

y_true = test["result"].values

# ── 1. Log-loss (primary metric) ─────────────────────────────────────
label_to_idx = {c: i for i, c in enumerate(CLASSES)}

def multiclass_log_loss(y_true, y_proba):
    indices = np.array([label_to_idx[y] for y in y_true])
    p = y_proba[np.arange(len(y_true)), indices]
    return -np.mean(np.log(np.maximum(p, 1e-15)))

ll_poisson = multiclass_log_loss(y_true, poisson_proba)
ll_elo = multiclass_log_loss(y_true, elo_proba)

print("-- Log-loss (lower is better) ----------------------")
print(f"  Poisson + DC:   {ll_poisson:.4f}")
print(f"  Elo baseline:   {ll_elo:.4f}")
print(f"  Improvement:    {(ll_elo - ll_poisson):.4f}  ({(ll_elo - ll_poisson) / ll_elo * 100:.1f}%)\n")

# ── 2. Draw rate ─────────────────────────────────────────────────────
classes_arr = np.array(CLASSES)
pred_class_poisson = classes_arr[np.argmax(poisson_proba, axis=1)]
pred_class_elo = classes_arr[np.argmax(elo_proba, axis=1)]

actual_draw = (y_true == "D").mean()
pred_draw_poisson = (pred_class_poisson == "D").mean()
pred_draw_elo = (pred_class_elo == "D").mean()

print("-- Draw rate (target: ~23.1%) ----------------------")
print(f"  Actual:         {actual_draw:.1%}")
print(f"  Poisson + DC:   {pred_draw_poisson:.1%}")
print(f"  Elo baseline:   {pred_draw_elo:.1%}\n")

# ── 3. Accuracy ──────────────────────────────────────────────────────
acc_poisson = accuracy_score(y_true, pred_class_poisson)
acc_elo = accuracy_score(y_true, pred_class_elo)

print("-- Accuracy ----------------------------------------")
print(f"  Poisson + DC:   {acc_poisson:.3f}")
print(f"  Elo baseline:   {acc_elo:.3f}\n")

# ── 4. Probability calibration by actual outcome ─────────────────────
test_copy = test.copy()
test_copy[["p_H", "p_D", "p_A"]] = poisson_proba

cal = (
    test_copy.groupby("result")
    .agg(
        count=("result", "size"),
        mean_p_H=("p_H", "mean"),
        mean_p_D=("p_D", "mean"),
        mean_p_A=("p_A", "mean"),
    )
    .loc[CLASSES]
)
print("-- Mean predicted probabilities by actual result ---")
print("  (diagonal should be highest in each row)")
print(cal.to_string(float_format="{:.3f}".format))
print()

# ── 5. Draw predictions: where does the model predict draws? ─────────
draw_pred_mask = np.array(pred_class_poisson) == "D"
if draw_pred_mask.sum() > 0:
    draw_preds = test_copy[draw_pred_mask][["date", "home_team", "away_team",
                                            "result", "p_H", "p_D", "p_A"]].head(10)
    print("-- Sample matches predicted as Draw (Poisson+DC) ---")
    print(draw_preds.to_string(index=False, float_format="{:.3f}".format))
else:
    print("-- No matches predicted as Draw (model still avoiding D) --")
