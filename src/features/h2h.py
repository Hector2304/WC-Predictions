"""
Head-to-head goal statistics with Bayesian shrinkage.

For each match, computes the directional H2H mean (home_team vs this
specific away_team) using only matches BEFORE that date. Shrinks toward
the global prior when few H2H matches are available.

Direction matters: Argentina-Bolivia at home is a separate entry from
Bolivia-Argentina at home.
"""

import pandas as pd
from typing import Dict, List, Tuple


def compute_h2h(
    df: pd.DataFrame,
    global_home_avg: float,
    global_away_avg: float,
    k: float = 5.0,
) -> pd.DataFrame:
    """
    df must be sorted chronologically (ascending date).

    global_home_avg / global_away_avg: shrinkage priors — compute from
    the training set only to avoid leakage.
    k: prior strength in equivalent number of matches. k=5 means
    the first real H2H match moves the estimate by 1/(1+5) = 17%.
    """
    history: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    home_mu: List[float] = []
    away_mu: List[float] = []
    n_seen: List[int] = []

    for row in df.itertuples(index=False):
        key = (row.home_team, row.away_team)
        past = history.get(key, [])
        n = len(past)

        if n == 0:
            home_mu.append(global_home_avg)
            away_mu.append(global_away_avg)
        else:
            sum_h = sum(g[0] for g in past)
            sum_a = sum(g[1] for g in past)
            home_mu.append((sum_h + k * global_home_avg) / (n + k))
            away_mu.append((sum_a + k * global_away_avg) / (n + k))

        n_seen.append(n)
        history.setdefault(key, []).append((row.home_score, row.away_score))

    out = df.copy()
    out["h2h_home_goals_mu"] = home_mu
    out["h2h_away_goals_mu"] = away_mu
    out["h2h_n"] = n_seen
    return out
