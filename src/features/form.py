"""
Recent form features for each team over a rolling window.

Tracks (goals_scored, goals_conceded) from each team's own perspective
regardless of home/away. Shrinks to global mean when history is short.

Features added:
  form_home_attack   — home team's recent avg goals scored
  form_home_defense  — home team's recent avg goals conceded
  form_away_attack   — away team's recent avg goals scored
  form_away_defense  — away team's recent avg goals conceded
"""

import pandas as pd
from collections import deque
from typing import Dict, Deque, List, Tuple


def compute_form(
    df: pd.DataFrame,
    global_avg: float,
    window: int = 5,
    k: float = 3.0,
) -> pd.DataFrame:
    """
    df must be sorted chronologically.

    global_avg: (global_home_avg + global_away_avg) / 2 — average goals
    per team per match; used as shrinkage target for both attack and defense.
    window: rolling window size.
    k: prior strength in equivalent matches.
    """
    # (goals_scored, goals_conceded) from each team's perspective
    form: Dict[str, Deque[Tuple[int, int]]] = {}
    ha: List[float] = []
    hd: List[float] = []
    aa: List[float] = []
    ad: List[float] = []

    for row in df.itertuples(index=False):
        ht, at = row.home_team, row.away_team
        h_hist = list(form.get(ht, []))
        a_hist = list(form.get(at, []))

        ha.append(_shrunken([g[0] for g in h_hist], global_avg, k))
        hd.append(_shrunken([g[1] for g in h_hist], global_avg, k))
        aa.append(_shrunken([g[0] for g in a_hist], global_avg, k))
        ad.append(_shrunken([g[1] for g in a_hist], global_avg, k))

        # Update AFTER recording (no leakage)
        if ht not in form:
            form[ht] = deque(maxlen=window)
        if at not in form:
            form[at] = deque(maxlen=window)

        form[ht].append((row.home_score, row.away_score))
        # away team's perspective: they scored away_score, conceded home_score
        form[at].append((row.away_score, row.home_score))

    out = df.copy()
    out["form_home_attack"] = ha
    out["form_home_defense"] = hd
    out["form_away_attack"] = aa
    out["form_away_defense"] = ad
    return out


def _shrunken(values: List[int], prior: float, k: float) -> float:
    n = len(values)
    if n == 0:
        return prior
    return (sum(values) + k * prior) / (n + k)
