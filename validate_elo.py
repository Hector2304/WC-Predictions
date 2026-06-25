"""
Validate Elo ratings on three questions:
  1. Does it beat "always predict home" and random baselines?
  2. Does home advantage show up in the raw data?
  3. Is the model calibrated? (higher Elo diff → higher actual win rate)

Run from the project root:
    python validate_elo.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.data.load import load_results
from src.features.elo import HOME_ADVANTAGE, compute_elo

TEST_START = "2023-01-01"


def label_result(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def elo_predict(elo_diff: float, neutral: bool) -> str:
    """Predict winner from pre-match Elo difference."""
    adj = elo_diff if neutral else elo_diff + HOME_ADVANTAGE
    if adj > 0:
        return "H"
    if adj < 0:
        return "A"
    return "D"


# ── Load & compute ───────────────────────────────────────────────────
df_raw = load_results()
df, final_ratings = compute_elo(df_raw)
df["result"] = df.apply(lambda r: label_result(r.home_score, r.away_score), axis=1)

# ── Temporal split ───────────────────────────────────────────────────
test = df[df["date"] >= TEST_START].copy()
print(f"Total matches: {len(df):,}  |  Test set (>= {TEST_START}): {len(test):,}\n")

# ── 1. Accuracy vs baselines ─────────────────────────────────────────
test["elo_pred"] = test.apply(
    lambda r: elo_predict(r.elo_diff, r.neutral), axis=1
)

elo_acc = accuracy_score(test["result"], test["elo_pred"])
home_acc = accuracy_score(test["result"], ["H"] * len(test))
away_acc = accuracy_score(test["result"], ["A"] * len(test))
draw_acc = accuracy_score(test["result"], ["D"] * len(test))

print("-- Accuracy (test set) -----------------------------")
print(f"  Elo model:        {elo_acc:.3f}")
print(f"  Always home:      {home_acc:.3f}")
print(f"  Always away:      {away_acc:.3f}")
print(f"  Always draw:      {draw_acc:.3f}")
print(f"  Random (1/3):     {1/3:.3f}\n")

# -- 2. Home advantage in the raw data --------------------------------
non_neutral = df[~df["neutral"]]
hw = (non_neutral["result"] == "H").mean()
dw = (non_neutral["result"] == "D").mean()
aw = (non_neutral["result"] == "A").mean()

print("-- Result distribution (non-neutral, full dataset) -")
print(f"  Home wins:   {hw:.1%}")
print(f"  Draws:       {dw:.1%}")
print(f"  Away wins:   {aw:.1%}\n")

# -- 3. Calibration by Elo difference ---------------------------------
test_nn = test[~test["neutral"]].copy()
test_nn["elo_diff_adj"] = test_nn["elo_diff"] + HOME_ADVANTAGE

bins = [-np.inf, -200, -100, -50, 0, 50, 100, 200, np.inf]
labels = ["< -200", "-200/-100", "-100/-50", "-50/0", "0/50", "50/100", "100/200", "> 200"]
test_nn["bucket"] = pd.cut(test_nn["elo_diff_adj"], bins=bins, labels=labels)

cal = (
    test_nn.groupby("bucket", observed=True)["result"]
    .agg(
        matches="count",
        home_win_rate=lambda x: (x == "H").mean(),
        draw_rate=lambda x: (x == "D").mean(),
        away_win_rate=lambda x: (x == "A").mean(),
    )
    .reset_index()
)

print("-- Calibration: Elo diff bucket vs actual outcome rate --")
print(cal.to_string(index=False))
print()

# -- 4. Top / bottom rated teams --------------------------------------
ratings_df = (
    pd.Series(final_ratings, name="elo")
    .sort_values(ascending=False)
    .rename_axis("team")
    .reset_index()
)

print("-- Top 20 teams (current Elo) ----------------------")
print(ratings_df.head(20).to_string(index=False))
print()
print("-- Bottom 10 teams ---------------------------------")
print(ratings_df.tail(10).to_string(index=False))
