import math
from typing import Tuple

import numpy as np
import pandas as pd


# ── v1: attack-only form ──────────────────────────────────────────────────────

def get_wc_form(df: pd.DataFrame, year: int = 2026) -> Tuple[dict, float]:
    """
    Raw tournament form: goals scored and conceded per game in WC {year}.

    Returns:
        team_stats: dict[team -> {games, scored, conceded, gd, gpg, cpg}]
        tournament_avg_gpg: goals scored per team per game across the whole WC
    """
    wc = df[
        (df["date"].dt.year == year)
        & (df["tournament"].str.lower() == "fifa world cup")
        & (df["home_score"].notna())
        & (df["away_score"].notna())
    ]

    team_stats: dict = {}

    for _, row in wc.iterrows():
        for team, scored, conceded in [
            (row["home_team"], row["home_score"], row["away_score"]),
            (row["away_team"], row["away_score"], row["home_score"]),
        ]:
            if team not in team_stats:
                team_stats[team] = {"games": 0, "scored": 0, "conceded": 0}
            team_stats[team]["games"] += 1
            team_stats[team]["scored"]   += int(scored)
            team_stats[team]["conceded"] += int(conceded)

    for s in team_stats.values():
        g = s["games"]
        s["gd"]  = s["scored"] - s["conceded"]
        s["gpg"] = s["scored"]   / g if g > 0 else 0.0
        s["cpg"] = s["conceded"] / g if g > 0 else 0.0

    if wc.empty:
        tournament_avg_gpg = 1.5
    else:
        total_goals = float(wc["home_score"].sum() + wc["away_score"].sum())
        tournament_avg_gpg = total_goals / (2 * len(wc))

    return team_stats, tournament_avg_gpg


def form_adjusted_lambdas(
    lam_h: float,
    lam_a: float,
    home_team: str,
    away_team: str,
    team_stats: dict,
    tournament_avg_gpg: float,
    alpha: float = 0.7,
) -> Tuple[float, float]:
    """
    Attack-only form blend (v1).
    lambda_adj = lambda_base * (alpha + (1-alpha) * team_gpg/avg_gpg)
    Teams with no WC data default to ratio=1.0 (no change).
    """
    def attack_ratio(team: str) -> float:
        s = team_stats.get(team)
        if s is None or s["games"] == 0 or tournament_avg_gpg == 0:
            return 1.0
        return s["gpg"] / tournament_avg_gpg

    return (
        lam_h * (alpha + (1.0 - alpha) * attack_ratio(home_team)),
        lam_a * (alpha + (1.0 - alpha) * attack_ratio(away_team)),
    )


# ── v2: attack + defense + opponent quality ───────────────────────────────────

def get_wc_form_v2(
    df: pd.DataFrame,
    final_ratings: dict,
    year: int = 2026,
) -> Tuple[dict, float, float]:
    """
    Quality-adjusted tournament form: weights each goal by the opponent's Elo.

    Attack (qa_gpg):  scoring vs strong opponents is worth more.
                      weight = opp_elo / avg_wc_elo
    Defense (qa_cpg): conceding vs weak opponents is penalized more.
                      weight = avg_wc_elo / opp_elo

    Returns:
        team_stats: dict[team -> {games, qa_gpg, qa_cpg}]
        avg_qa_gpg: tournament-wide average quality-adjusted goals scored per game
        avg_qa_cpg: tournament-wide average quality-adjusted goals conceded per game
    """
    wc = df[
        (df["date"].dt.year == year)
        & (df["tournament"].str.lower() == "fifa world cup")
        & (df["home_score"].notna())
        & (df["away_score"].notna())
    ]

    wc_teams = set(wc["home_team"].tolist()) | set(wc["away_team"].tolist())
    avg_elo = (
        float(np.mean([final_ratings.get(t, 1500) for t in wc_teams]))
        if wc_teams else 1500.0
    )

    acc: dict = {}

    for _, row in wc.iterrows():
        h, a  = row["home_team"], row["away_team"]
        hs    = float(row["home_score"])
        as_   = float(row["away_score"])
        h_elo = final_ratings.get(h, avg_elo)
        a_elo = final_ratings.get(a, avg_elo)

        for team, scored, conceded, opp_elo in [
            (h, hs, as_, a_elo),
            (a, as_, hs, h_elo),
        ]:
            if team not in acc:
                acc[team] = {"games": 0, "scored_w": 0.0, "conceded_w": 0.0}
            acc[team]["games"] += 1
            acc[team]["scored_w"]   += scored   * (opp_elo  / avg_elo)
            acc[team]["conceded_w"] += conceded * (avg_elo  / opp_elo)

    team_stats: dict = {}
    for team, r in acc.items():
        g = r["games"]
        team_stats[team] = {
            "games":  g,
            "qa_gpg": r["scored_w"]   / g if g > 0 else 0.0,
            "qa_cpg": r["conceded_w"] / g if g > 0 else 0.0,
        }

    if not team_stats:
        return team_stats, 1.5, 1.5

    vals_gpg = [s["qa_gpg"] for s in team_stats.values()]
    vals_cpg = [s["qa_cpg"] for s in team_stats.values()]
    return team_stats, float(np.mean(vals_gpg)), float(np.mean(vals_cpg))


def form_adjusted_lambdas_v2(
    lam_h: float,
    lam_a: float,
    home_team: str,
    away_team: str,
    team_stats_v2: dict,
    avg_qa_gpg: float,
    avg_qa_cpg: float,
    alpha: float = 0.7,
) -> Tuple[float, float]:
    """
    Attack + defense + opponent-quality form blend (v2).

    For home goals:  geometric mean of (home attack ratio, away defense ratio)
    For away goals:  geometric mean of (away attack ratio, home defense ratio)

    home_attack_ratio  = home qa_gpg / avg_qa_gpg  (> 1 → strong attack vs quality)
    away_defense_ratio = away qa_cpg / avg_qa_cpg  (> 1 → leaky defense vs quality)

    lambda_h_adj = lambda_h * (alpha + (1-alpha) * sqrt(home_atk * away_def))
    """
    def get_ratios(team: str) -> Tuple[float, float]:
        s = team_stats_v2.get(team)
        if s is None or s["games"] == 0:
            return 1.0, 1.0
        atk = s["qa_gpg"] / avg_qa_gpg if avg_qa_gpg > 0 else 1.0
        dfs = s["qa_cpg"] / avg_qa_cpg if avg_qa_cpg > 0 else 1.0
        return atk, dfs

    home_atk, home_def = get_ratios(home_team)
    away_atk, away_def = get_ratios(away_team)

    home_combined = math.sqrt(home_atk * away_def)
    away_combined = math.sqrt(away_atk * home_def)

    return (
        lam_h * (alpha + (1.0 - alpha) * home_combined),
        lam_a * (alpha + (1.0 - alpha) * away_combined),
    )


# ── v3: form_v2 + FIFA advanced stats ─────────────────────────────────────────

def form_adjusted_lambdas_v3(
    lam_h: float,
    lam_a: float,
    home_team: str,
    away_team: str,
    team_stats_v2: dict,
    avg_qa_gpg: float,
    avg_qa_cpg: float,
    fifa_data: dict,
    alpha_form: float = 0.70,
    alpha_fifa: float = 0.90,
) -> Tuple[float, float]:
    """
    Form v3: quality-adjusted tournament form (v2) + FIFA advanced stats.

    Chains two adjustments on the v5 base lambdas:
      1. form_v2 (quality-adjusted goals, 70/30 blend)
      2. FIFA layer (attack/defense factors, 90/10 blend)

    FIFA combined factor for home goals:
      sqrt(home_attack_factor / away_defense_factor)
      > 1 → home attack stronger than away defense → lambda increases
      < 1 → home attack weaker than away defense → lambda decreases

    Teams absent from fifa_data receive a factor of 1.0 (no change).
    Combined factor is clamped to [0.5, 2.0].
    """
    lam_h_v2, lam_a_v2 = form_adjusted_lambdas_v2(
        lam_h, lam_a, home_team, away_team,
        team_stats_v2, avg_qa_gpg, avg_qa_cpg, alpha=alpha_form,
    )

    _neutral = {"attack_factor": 1.0, "defense_factor": 1.0}
    home_fifa = fifa_data.get(home_team, _neutral)
    away_fifa = fifa_data.get(away_team, _neutral)

    home_combined = max(0.5, min(2.0, math.sqrt(
        home_fifa["attack_factor"] / away_fifa["defense_factor"]
    )))
    away_combined = max(0.5, min(2.0, math.sqrt(
        away_fifa["attack_factor"] / home_fifa["defense_factor"]
    )))

    return (
        lam_h_v2 * (alpha_fifa + (1.0 - alpha_fifa) * home_combined),
        lam_a_v2 * (alpha_fifa + (1.0 - alpha_fifa) * away_combined),
    )
