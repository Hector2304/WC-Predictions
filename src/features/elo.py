import pandas as pd
from typing import Dict, Tuple

INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 100.0  # Elo points added to home side for expected-score calc only

# Partial keyword → K-factor (checked in order, first match wins)
_K_RULES: list[tuple[str, float]] = [
    ("world cup", 60.0),
    ("copa america", 50.0),
    ("copa américa", 50.0),
    ("uefa euro", 50.0),
    ("african cup", 50.0),
    ("africa cup", 50.0),
    ("asian cup", 50.0),
    ("gold cup", 50.0),
    ("nations league", 45.0),
    ("qualification", 40.0),
    ("qualifier", 40.0),
    ("friendly", 20.0),
]
_DEFAULT_K = 30.0


def _k_factor(tournament: str) -> float:
    t = tournament.lower()
    for keyword, k in _K_RULES:
        if keyword in t:
            return k
    return _DEFAULT_K


def _expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def compute_elo(
    df: pd.DataFrame,
    home_advantage: float = HOME_ADVANTAGE,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Traverse matches in chronological order (df must be pre-sorted).
    For each match:
      1. Read current ratings  → store as pre-match snapshot (no leakage)
      2. Compute expected scores with home-advantage adjustment
      3. Update ratings based on actual result

    home_advantage: Elo points added to home team for E[score] calc only.
    Sweep this value to find the best calibration.

    Returns df with added columns:
      home_elo_pre, away_elo_pre, elo_diff (home - away, pre-match)
    and a dict of final ratings keyed by team name.
    """
    ratings: Dict[str, float] = {}
    home_pre: list[float] = []
    away_pre: list[float] = []

    for row in df.itertuples(index=False):
        r_h = ratings.get(row.home_team, INITIAL_RATING)
        r_a = ratings.get(row.away_team, INITIAL_RATING)

        home_pre.append(r_h)
        away_pre.append(r_a)

        r_h_adj = r_h if row.neutral else r_h + home_advantage
        e_h = _expected(r_h_adj, r_a)
        e_a = 1.0 - e_h

        if row.home_score > row.away_score:
            s_h, s_a = 1.0, 0.0
        elif row.home_score < row.away_score:
            s_h, s_a = 0.0, 1.0
        else:
            s_h = s_a = 0.5

        k = _k_factor(row.tournament)
        ratings[row.home_team] = r_h + k * (s_h - e_h)
        ratings[row.away_team] = r_a + k * (s_a - e_a)

    out = df.copy()
    out["home_elo_pre"] = home_pre
    out["away_elo_pre"] = away_pre
    out["elo_diff"] = out["home_elo_pre"] - out["away_elo_pre"]
    return out, ratings
