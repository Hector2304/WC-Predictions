"""
Error analysis on the Elo model — three diagnostic questions:
  1. Confusion matrix: where do the mistakes concentrate (H/D/A)?
  2. Accuracy by |elo_diff| bucket: is the model failing on close matches?
  3. Accuracy by tournament type: systematic bias by competition?

Run from the project root:
    py error_analysis.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

from src.data.load import load_results
from src.features.elo import HOME_ADVANTAGE, compute_elo

TEST_START = "2023-01-01"
CLASSES = ["H", "D", "A"]


def label_result(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def elo_predict(elo_diff: float, neutral: bool) -> str:
    adj = elo_diff if neutral else elo_diff + HOME_ADVANTAGE
    if adj > 0:
        return "H"
    if adj < 0:
        return "A"
    return "D"


def tournament_category(tournament: str) -> str:
    t = tournament.lower()
    if "world cup" in t and "qual" not in t:
        return "World Cup"
    if "qual" in t or "qualifier" in t:
        return "Qualifier"
    if any(x in t for x in [
        "copa america", "copa am", "uefa euro", "african cup",
        "africa cup", "asian cup", "gold cup", "nations league",
    ]):
        return "Continental"
    if "friendly" in t:
        return "Friendly"
    return "Other"


# ── Load & compute ────────────────────────────────────────────────────
df_raw = load_results()
df, _ = compute_elo(df_raw)
df["result"] = df.apply(lambda r: label_result(r.home_score, r.away_score), axis=1)

test = df[df["date"] >= TEST_START].copy()
test["elo_pred"] = test.apply(lambda r: elo_predict(r.elo_diff, r.neutral), axis=1)
test["elo_diff_adj"] = test.apply(
    lambda r: r.elo_diff if r.neutral else r.elo_diff + HOME_ADVANTAGE, axis=1
)
test["abs_elo_diff"] = test["elo_diff_adj"].abs()
test["tournament_cat"] = test["tournament"].apply(tournament_category)

print(f"Test set: {len(test):,} matches ({TEST_START} onwards)\n")

# ── 1. Confusion matrix ───────────────────────────────────────────────
cm = confusion_matrix(test["result"], test["elo_pred"], labels=CLASSES)
cm_df = pd.DataFrame(cm, index=[f"actual {c}" for c in CLASSES],
                     columns=[f"pred {c}" for c in CLASSES])

# Normalized by actual row (shows where each true class goes)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm_df = pd.DataFrame(
    np.round(cm_norm, 3),
    index=[f"actual {c}" for c in CLASSES],
    columns=[f"pred {c}" for c in CLASSES],
)

print("-- Confusion matrix (counts) -----------------------")
print(cm_df.to_string())
print()
print("-- Confusion matrix (row-normalized) ---------------")
print(cm_norm_df.to_string())
print()

# Draw prediction summary
pred_draw_rate = (test["elo_pred"] == "D").mean()
actual_draw_rate = (test["result"] == "D").mean()
print(f"  Actual draw rate:    {actual_draw_rate:.1%}")
print(f"  Predicted draw rate: {pred_draw_rate:.1%}")
print()

# ── 2. Accuracy by |elo_diff| bucket ─────────────────────────────────
bins = [0, 100, 300, np.inf]
labels_bucket = ["< 100  (toss-up)", "100-300 (clear fav)", "> 300  (dominant)"]
test["elo_bucket"] = pd.cut(test["abs_elo_diff"], bins=bins, labels=labels_bucket)

bucket_stats = (
    test.groupby("elo_bucket", observed=True)
    .apply(
        lambda g: pd.Series({
            "matches": len(g),
            "accuracy": accuracy_score(g["result"], g["elo_pred"]),
            "draw_rate_actual": (g["result"] == "D").mean(),
            "draw_rate_pred": (g["elo_pred"] == "D").mean(),
        })
    )
    .reset_index()
)

print("-- Accuracy by |elo_diff| bucket -------------------")
print(bucket_stats.to_string(index=False, float_format="{:.3f}".format))
print()

# ── 3. Accuracy by tournament type ───────────────────────────────────
tourn_stats = (
    test.groupby("tournament_cat")
    .apply(
        lambda g: pd.Series({
            "matches": len(g),
            "accuracy": accuracy_score(g["result"], g["elo_pred"]),
            "draw_rate_actual": (g["result"] == "D").mean(),
            "home_win_rate_actual": (g["result"] == "H").mean(),
        })
    )
    .sort_values("matches", ascending=False)
    .reset_index()
)

print("-- Accuracy by tournament category -----------------")
print(tourn_stats.to_string(index=False, float_format="{:.3f}".format))
print()

# ── 4. Biggest misses: matches the model was most wrong about ─────────
test["correct"] = test["result"] == test["elo_pred"]
wrong = test[~test["correct"]].copy()

# Among wrong predictions, where was the model most confident (highest |adj_diff|)?
worst = (
    wrong.nlargest(10, "abs_elo_diff")[
        ["date", "home_team", "away_team", "home_score", "away_score",
         "tournament_cat", "elo_diff_adj", "result", "elo_pred"]
    ]
)

print("-- Top 10 biggest upsets (model was most confident, still wrong) --")
print(worst.to_string(index=False))
