"""
Calibrate decision threshold theta_D for draw predictions.

Split:
  train  : < 2022          (fits PoissonDC)
  val    : 2022            (calibrates theta_D)
  test   : >= 2023         (final evaluation, untouched during calibration)

Decision rule:
  if P(D) > theta_D  →  predict "D"
  else               →  argmax(P(H), P(A))

theta_D is chosen to maximise macro F1 on the validation set.
Argmax is equivalent to theta_D = max(P(H), P(A)) per row, which in
practice never predicts D (as confirmed in error_analysis.py).

Run from the project root:
    py calibrate_threshold.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.data.load import load_results
from src.features.elo import compute_elo
from src.features.poisson import PoissonDC

TRAIN_END = "2022-01-01"
VAL_END = "2023-01-01"
CLASSES = ["H", "D", "A"]


# ── helpers ───────────────────────────────────────────────────────────

def label_result(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def apply_threshold(proba: np.ndarray, theta_d: float) -> np.ndarray:
    """
    Predict D when P(D) > theta_d; otherwise argmax between H and A.
    """
    p_h, p_d, p_a = proba[:, 0], proba[:, 1], proba[:, 2]
    ha = np.where(p_h >= p_a, "H", "A")
    return np.where(p_d > theta_d, "D", ha)


def multiclass_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx = np.array([label_to_idx[y] for y in y_true])
    p = y_proba[np.arange(len(y_true)), idx]
    return float(-np.mean(np.log(np.maximum(p, 1e-15))))


# ── load, elo, split ──────────────────────────────────────────────────
df_raw = load_results()
df, _ = compute_elo(df_raw)
df["result"] = df.apply(lambda r: label_result(r.home_score, r.away_score), axis=1)

train = df[df["date"] < TRAIN_END].copy()
val   = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)].copy()
test  = df[df["date"] >= VAL_END].copy()

print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}\n")

# ── fit ───────────────────────────────────────────────────────────────
model = PoissonDC(alpha=0.0)
model.fit(train)
print(f"rho = {model.rho_:.4f}\n")

val_proba  = model.predict_proba(val)
test_proba = model.predict_proba(test)

# ── calibrate theta_D on val ─────────────────────────────────────────
thresholds = np.round(np.arange(0.10, 0.46, 0.01), 2)
rows = []
for theta in thresholds:
    pred = apply_threshold(val_proba, theta)
    rows.append({
        "theta_D": theta,
        "accuracy": accuracy_score(val["result"], pred),
        "f1_macro": f1_score(val["result"], pred, labels=CLASSES, average="macro", zero_division=0),
        "f1_draw":  f1_score(val["result"], pred, labels=["D"], average="macro", zero_division=0),
        "draw_rate_pred": (pred == "D").mean(),
    })

cal_df = pd.DataFrame(rows)
best_idx = cal_df["f1_macro"].idxmax()
best = cal_df.loc[best_idx]
theta_best = best["theta_D"]

print("-- Threshold search on val (2022) ----------------")
print(cal_df.to_string(index=False, float_format="{:.4f}".format))
print()
print(f"Best theta_D = {theta_best:.2f}  |  F1_macro = {best['f1_macro']:.4f}"
      f"  |  draw_rate_pred = {best['draw_rate_pred']:.1%}\n")

# ── evaluate on test ──────────────────────────────────────────────────
y_test = test["result"].values
argmax_pred = np.array(CLASSES)[np.argmax(test_proba, axis=1)]
thresh_pred = apply_threshold(test_proba, theta_best)

print("-- Test set comparison (argmax vs calibrated threshold) --")
print(f"{'Metric':<28} {'Argmax':>10} {'Threshold':>12}")
print(f"{'Accuracy':<28} {accuracy_score(y_test, argmax_pred):>10.3f} {accuracy_score(y_test, thresh_pred):>12.3f}")
print(f"{'F1 macro':<28} {f1_score(y_test, argmax_pred, labels=CLASSES, average='macro', zero_division=0):>10.3f} {f1_score(y_test, thresh_pred, labels=CLASSES, average='macro', zero_division=0):>12.3f}")
print(f"{'F1 draw':<28} {f1_score(y_test, argmax_pred, labels=['D'], average='macro', zero_division=0):>10.3f} {f1_score(y_test, thresh_pred, labels=['D'], average='macro', zero_division=0):>12.3f}")
print(f"{'Draw rate predicted':<28} {(argmax_pred=='D').mean():>10.1%} {(thresh_pred=='D').mean():>12.1%}")
print(f"{'Draw rate actual':<28} {(y_test=='D').mean():>10.1%} {(y_test=='D').mean():>12.1%}")
print(f"{'Log-loss':<28} {multiclass_log_loss(y_test, test_proba):>10.4f} {'(same)':>12}")
print()

print("-- Per-class F1 on test (calibrated threshold) ---")
print(classification_report(y_test, thresh_pred, labels=CLASSES, digits=3, zero_division=0))

# ── sample: what does the model predict as draws? ─────────────────────
draw_mask = thresh_pred == "D"
if draw_mask.sum() > 0:
    test_copy = test.copy()
    test_copy[["p_H", "p_D", "p_A"]] = test_proba
    scores = model.predict_scoreline(test_copy)
    test_copy["pred_score"] = scores["pred_home"].astype(str) + "-" + scores["pred_away"].astype(str)
    sample = (
        test_copy[draw_mask]
        [["date", "home_team", "away_team", "home_score", "away_score",
          "pred_score", "p_H", "p_D", "p_A"]]
        .head(15)
    )
    print(f"-- Sample draw predictions on test (n={draw_mask.sum()}) ---")
    print(sample.to_string(index=False, float_format="{:.3f}".format))
else:
    print("-- No draw predictions even with threshold — theta_D may be too high.")
