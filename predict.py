"""
Predict match outcomes using the v5 model.

The pipeline recomputes Elo and H2H from the full dataset (including the
2026 WC group stage), so ratings reflect the most current state.
All World Cup matches are neutral=True by default.

Usage:
    # Group-stage match (default)
    py predict.py "Argentina" "France"

    # Knockout match (uses theta_D_knockout=0.28 instead of theta_D=0.26)
    py predict.py "Argentina" "France" --knockout

    # Multiple matches
    py predict.py "Spain" "Germany" "Brazil" "England" --knockout

    # From file (one pair per line: Team A,Team B)
    py predict.py --file r16_fixtures.txt --knockout

    # List all teams with current Elo rating
    py predict.py --teams

    # Non-neutral (e.g. hypothetical home match)
    py predict.py "Brazil" "Argentina" --not-neutral
"""

import argparse
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.data.load import load_results
from src.features.elo import compute_elo
from src.features.h2h import compute_h2h
from src.features.tournament import add_tournament_features
from src.features.tournament_form import (
    form_adjusted_lambdas,
    form_adjusted_lambdas_v2,
    get_wc_form,
    get_wc_form_v2,
)

CLASSES = ["H", "D", "A"]


# ── pipeline state ────────────────────────────────────────────────────────────

def load_pipeline():
    with open("models/v5_config.json") as f:
        cfg = json.load(f)
    model = joblib.load("models/poisson_dc_v5.joblib")
    return model, cfg


def build_current_state(cfg):
    ha = cfg.get("home_advantage", 20.0)
    p  = cfg["priors"]
    hp = cfg["hyperparams"]

    df_raw = load_results()
    df, final_ratings = compute_elo(df_raw, home_advantage=ha)
    df = compute_h2h(df, p["global_home_avg"], p["global_away_avg"], k=hp["h2h_k"])
    df = add_tournament_features(df)
    return df, final_ratings


# ── prediction logic ──────────────────────────────────────────────────────────

def predict_match(
    home_team: str,
    away_team: str,
    neutral: bool,
    knockout: bool,
    model,
    cfg: dict,
    final_ratings: dict,
    df_full: pd.DataFrame,
    form_data=None,     # (team_stats, avg_gpg) — attack-only (v1)
    form_data_v2=None,  # (team_stats_v2, avg_qa_gpg, avg_qa_cpg) — full (v2)
    form_alpha: float = 0.7,
) -> dict:
    p = cfg["priors"]
    k = cfg["hyperparams"]["h2h_k"]

    r_h = final_ratings.get(home_team)
    r_a = final_ratings.get(away_team)
    if r_h is None:
        raise ValueError(f"Unknown team: '{home_team}'. Run --teams to see valid names.")
    if r_a is None:
        raise ValueError(f"Unknown team: '{away_team}'. Run --teams to see valid names.")

    # For neutral knockout matches normalize to higher-Elo team as "home" so
    # the directional H2H lookup is deterministic regardless of file/input order.
    h2h_swapped = False
    if knockout and neutral and r_a > r_h:
        home_team, away_team = away_team, home_team
        r_h, r_a = r_a, r_h
        h2h_swapped = True

    elo_diff = r_h - r_a

    mask     = (df_full["home_team"] == home_team) & (df_full["away_team"] == away_team)
    h2h_hist = df_full[mask]
    n_h2h    = len(h2h_hist)

    if n_h2h == 0:
        h2h_home = p["global_home_avg"]
        h2h_away = p["global_away_avg"]
    else:
        h2h_home = (h2h_hist["home_score"].sum() + k * p["global_home_avg"]) / (n_h2h + k)
        h2h_away = (h2h_hist["away_score"].sum() + k * p["global_away_avg"]) / (n_h2h + k)

    feature_row = pd.DataFrame([{
        "elo_diff":          elo_diff,
        "neutral":           float(neutral),
        "h2h_home_goals_mu": h2h_home,
        "h2h_away_goals_mu": h2h_away,
        "is_world_cup":      1.0,
        "is_knockout":       1.0 if knockout else 0.0,
    }])

    theta_D   = cfg["theta_D_knockout"] if knockout else cfg["theta_D"]
    lam_h_arr, lam_a_arr = model.predict_lambdas(feature_row)
    proba     = model.predict_proba(feature_row)
    scoreline = model.predict_scoreline(feature_row)

    p_h, p_d, p_a = float(proba[0, 0]), float(proba[0, 1]), float(proba[0, 2])
    if p_d > theta_D:
        prediction = "D"
    elif p_h >= p_a:
        prediction = "H"
    else:
        prediction = "A"

    # ── form adjustment (optional) ────────────────────────────────────────────
    form_result = None
    if form_data is not None:
        team_stats, tournament_avg_gpg = form_data
        lam_h_adj, lam_a_adj = form_adjusted_lambdas(
            float(lam_h_arr[0]), float(lam_a_arr[0]),
            home_team, away_team,
            team_stats, tournament_avg_gpg,
            alpha=form_alpha,
        )
        proba_adj, sh_adj, sa_adj = model.predict_from_lambdas(
            np.array([lam_h_adj]), np.array([lam_a_adj])
        )
        ph_adj = float(proba_adj[0, 0])
        pd_adj = float(proba_adj[0, 1])
        pa_adj = float(proba_adj[0, 2])
        if pd_adj > theta_D:
            pred_adj = "D"
        elif ph_adj >= pa_adj:
            pred_adj = "H"
        else:
            pred_adj = "A"

        def _team_form_entry(team):
            s = team_stats.get(team, {})
            return {
                "games":    s.get("games", 0),
                "scored":   s.get("scored", 0),
                "conceded": s.get("conceded", 0),
                "gd":       s.get("gd", 0),
                "gpg":      s.get("gpg", 0.0),
                "cpg":      s.get("cpg", 0.0),
            }

        form_result = {
            "home_form":            _team_form_entry(home_team),
            "away_form":            _team_form_entry(away_team),
            "tournament_avg_gpg":   round(tournament_avg_gpg, 3),
            "alpha":                form_alpha,
            "P(H)":                 round(ph_adj, 3),
            "P(D)":                 round(pd_adj, 3),
            "P(A)":                 round(pa_adj, 3),
            "prediction":           pred_adj,
            "pred_score":           f"{int(sh_adj[0])}-{int(sa_adj[0])}",
        }

    # ── form v2: attack + defense + opponent quality ──────────────────────────
    form_result_v2 = None
    if form_data_v2 is not None:
        team_stats_v2, avg_qa_gpg, avg_qa_cpg = form_data_v2
        lam_h_v2, lam_a_v2 = form_adjusted_lambdas_v2(
            float(lam_h_arr[0]), float(lam_a_arr[0]),
            home_team, away_team,
            team_stats_v2, avg_qa_gpg, avg_qa_cpg,
            alpha=form_alpha,
        )
        proba_v2, sh_v2, sa_v2 = model.predict_from_lambdas(
            np.array([lam_h_v2]), np.array([lam_a_v2])
        )
        ph_v2 = float(proba_v2[0, 0])
        pd_v2 = float(proba_v2[0, 1])
        pa_v2 = float(proba_v2[0, 2])
        if pd_v2 > theta_D:
            pred_v2 = "D"
        elif ph_v2 >= pa_v2:
            pred_v2 = "H"
        else:
            pred_v2 = "A"

        def _v2_entry(team):
            s = team_stats_v2.get(team, {})
            return {
                "games":   s.get("games", 0),
                "qa_gpg":  s.get("qa_gpg", 0.0),
                "qa_cpg":  s.get("qa_cpg", 0.0),
            }

        form_result_v2 = {
            "home_form":    _v2_entry(home_team),
            "away_form":    _v2_entry(away_team),
            "avg_qa_gpg":   round(avg_qa_gpg, 3),
            "avg_qa_cpg":   round(avg_qa_cpg, 3),
            "alpha":        form_alpha,
            "P(H)":         round(ph_v2, 3),
            "P(D)":         round(pd_v2, 3),
            "P(A)":         round(pa_v2, 3),
            "prediction":   pred_v2,
            "pred_score":   f"{int(sh_v2[0])}-{int(sa_v2[0])}",
        }

    return {
        "home_team":   home_team,
        "away_team":   away_team,
        "neutral":     neutral,
        "knockout":    knockout,
        "h2h_swapped": h2h_swapped,
        "home_elo":    round(r_h, 0),
        "away_elo":    round(r_a, 0),
        "elo_diff":    round(elo_diff, 0),
        "h2h_n":       n_h2h,
        "P(H)":        round(p_h, 3),
        "P(D)":        round(p_d, 3),
        "P(A)":        round(p_a, 3),
        "pred_score":  f"{int(scoreline['pred_home'].iloc[0])}-{int(scoreline['pred_away'].iloc[0])}",
        "prediction":  prediction,
        "favourite":   (
            home_team if p_h > p_a else
            away_team if p_a > p_h else "EVEN"
        ),
        "theta_D_used": theta_D,
        "form":         form_result,
        "form_v2":      form_result_v2,
    }


def _pred_row(r: dict, source: str, ph: float, pd: float, pa: float,
              pred: str, score: str, base_pred: str) -> str:
    changed = pred != base_pred
    change = f" <- {base_pred}" if changed else ""
    return (
        f"  {source:<10} {r['home_team']:<22} {r['away_team']:<22}"
        f"  {ph:.3f}  {pd:.3f}  {pa:.3f}  {score:>5}  {pred:>4}{change}"
    )


def print_results(results: list[dict]):
    if not results:
        return

    knockout_mode = any(r["knockout"] for r in results)
    phase = "KNOCKOUT" if knockout_mode else "GROUP STAGE"
    theta = results[0]["theta_D_used"]
    has_form = any(r.get("form") for r in results)

    print()
    if not has_form:
        header = (
            f"  {'Home':<22} {'Away':<22} {'P(H)':>6} {'P(D)':>6} {'P(A)':>6}"
            f"  {'Score':>5}  {'Pred':>4}  {'Elo diff':>9}  {'H2H':>3}"
        )
        print(f"  [{phase}]")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in results:
            label_h = r["home_team"] + ("*" if r["favourite"] == r["home_team"] else "")
            label_a = r["away_team"] + ("*" if r["favourite"] == r["away_team"] else "")
            print(
                f"  {label_h:<22} {label_a:<22}"
                f"  {r['P(H)']:.3f}  {r['P(D)']:.3f}  {r['P(A)']:.3f}"
                f"  {r['pred_score']:>5}  {r['prediction']:>4}"
                f"  {r['elo_diff']:>+9.0f}  {r['h2h_n']:>3}"
            )
        print()
        print(f"  * = model favourite   theta_D = {theta:.2f}   H2H = historical matchups")
    else:
        # ── 3-column comparison ───────────────────────────────────────────────
        col_header = (
            f"  {'Model':<10} {'Home':<22} {'Away':<22}"
            f"  {'P(H)':>6} {'P(D)':>6} {'P(A)':>6}  {'Score':>5}  {'Pred':>4}"
        )
        print(f"  [{phase}] - model comparison (alpha={results[0]['form']['alpha']:.2f})")
        print(col_header)

        for r in results:
            print("  " + "-" * (len(col_header) - 2))
            # v5 base
            print(_pred_row(r, "v5", r["P(H)"], r["P(D)"], r["P(A)"],
                            r["prediction"], r["pred_score"], r["prediction"]))
            # v1: attack-only form
            if r.get("form"):
                f1 = r["form"]
                print(_pred_row(r, "atk", f1["P(H)"], f1["P(D)"], f1["P(A)"],
                                f1["prediction"], f1["pred_score"], r["prediction"]))
            # v2: attack + defense + quality
            if r.get("form_v2"):
                f2 = r["form_v2"]
                print(_pred_row(r, "atk+def+q", f2["P(H)"], f2["P(D)"], f2["P(A)"],
                                f2["prediction"], f2["pred_score"], r["prediction"]))

        print()
        print(f"  theta_D = {theta:.2f}   '<- v5=X' = prediction changed from base model")

        # ── team stats table ──────────────────────────────────────────────────
        f0     = next(r["form"]    for r in results if r.get("form"))
        f0_v2  = next((r["form_v2"] for r in results if r.get("form_v2")), None)
        avg_gpg    = f0["tournament_avg_gpg"]
        avg_qa_gpg = f0_v2["avg_qa_gpg"] if f0_v2 else None
        avg_qa_cpg = f0_v2["avg_qa_cpg"] if f0_v2 else None

        seen_v1 = {}
        seen_v2 = {}
        for r in results:
            if r.get("form"):
                for team, side in [(r["home_team"], "home"), (r["away_team"], "away")]:
                    if team not in seen_v1:
                        seen_v1[team] = r["form"][f"{side}_form"]
            if r.get("form_v2"):
                for team, side in [(r["home_team"], "home"), (r["away_team"], "away")]:
                    if team not in seen_v2:
                        seen_v2[team] = r["form_v2"][f"{side}_form"]

        print()
        if f0_v2:
            print(f"  {'Team':<22} {'GP':>3}  {'GF':>3}  {'GA':>3}  {'GD':>4}"
                  f"  {'gpg':>5} {'raw/avg':>8}"
                  f"  {'qa_gpg':>7} {'atk/avg':>8}"
                  f"  {'qa_cpg':>7} {'def/avg':>8}")
            print("  " + "-" * 90)
            for team in sorted(seen_v1):
                s1 = seen_v1.get(team, {})
                s2 = seen_v2.get(team, {})
                g = s1.get("games", 0)
                if g == 0:
                    print(f"  {team:<22} {'—':>3}  no data")
                    continue
                raw_r  = s1["gpg"] / avg_gpg         if avg_gpg    > 0 else 1.0
                qa_atk = s2.get("qa_gpg", 0.0) / avg_qa_gpg if avg_qa_gpg > 0 else 1.0
                qa_def = s2.get("qa_cpg", 0.0) / avg_qa_cpg if avg_qa_cpg > 0 else 1.0
                d_raw  = "^" if raw_r  > 1 else "v"
                d_atk  = "^" if qa_atk > 1 else "v"
                d_def  = "^" if qa_def > 1 else "v"
                print(
                    f"  {team:<22} {g:>3}  {s1['scored']:>3}  {s1['conceded']:>3}  {s1['gd']:>+4}"
                    f"  {s1['gpg']:>5.2f} {d_raw}{raw_r:>6.2f}x"
                    f"  {s2.get('qa_gpg', 0):>7.2f} {d_atk}{qa_atk:>6.2f}x"
                    f"  {s2.get('qa_cpg', 0):>7.2f} {d_def}{qa_def:>6.2f}x"
                )
        else:
            print(f"  {'Team':<22} {'GP':>3}  {'GF':>3}  {'GA':>3}  {'GD':>4}  {'gpg':>5}  {'vs avg':>7}")
            print("  " + "-" * 54)
            for team, s in sorted(seen_v1.items()):
                g = s.get("games", 0)
                if g == 0:
                    print(f"  {team:<22}   no data")
                    continue
                ratio = s["gpg"] / avg_gpg if avg_gpg > 0 else 1.0
                d = "^" if ratio > 1 else "v"
                print(
                    f"  {team:<22} {g:>3}  {s['scored']:>3}  {s['conceded']:>3}"
                    f"  {s['gd']:>+4}  {s['gpg']:>5.2f}  {d}{ratio:>5.2f}x"
                )

    swapped = [r for r in results if r.get("h2h_swapped")]
    if swapped:
        print()
        print("  (ko) teams reordered by Elo for H2H lookup:")
        for r in swapped:
            print(f"    {r['home_team']} vs {r['away_team']}  "
                  f"[Elo {r['home_elo']:.0f} vs {r['away_elo']:.0f}]")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="World Cup 2026 predictor (v5)")
    parser.add_argument("teams", nargs="*", help="Pairs of team names: 'TeamA' 'TeamB' ...")
    parser.add_argument("--knockout", action="store_true",
                        help="Knockout phase (uses theta_D_knockout=0.28)")
    parser.add_argument("--not-neutral", action="store_true",
                        help="Treat as non-neutral (home team has venue advantage)")
    parser.add_argument("--teams", dest="list_teams", action="store_true",
                        help="List all teams with current Elo rating")
    parser.add_argument("--file", metavar="FILE",
                        help="Read fixtures from file (one pair per line: Team A,Team B)")
    parser.add_argument("--with-form", action="store_true",
                        help="Show form-adjusted prediction alongside v5 (WC 2026 attack form)")
    parser.add_argument("--form-alpha", type=float, default=0.7, metavar="A",
                        help="Weight on base model in form blend (default: 0.70)")
    args = parser.parse_args()

    model, cfg = load_pipeline()
    df_full, final_ratings = build_current_state(cfg)

    form_data    = None
    form_data_v2 = None
    if args.with_form:
        form_data    = get_wc_form(df_full)
        form_data_v2 = get_wc_form_v2(df_full, final_ratings)

    if args.list_teams:
        ratings_df = (
            pd.Series(final_ratings, name="elo")
            .sort_values(ascending=False)
            .rename_axis("team")
            .reset_index()
        )
        ratings_df["elo"] = ratings_df["elo"].round(0).astype(int)
        print(f"\n  {'Rank':<6} {'Team':<30} {'Elo':>6}")
        print("  " + "-" * 44)
        for i, row in ratings_df.iterrows():
            print(f"  {i+1:<6} {row['team']:<30} {row['elo']:>6}")
        return

    matchups = []
    if args.file:
        with open(args.file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    matchups.append((parts[0], parts[1]))
                else:
                    print(f"  Skipping invalid line: {line}", file=sys.stderr)
    elif args.teams:
        if len(args.teams) % 2 != 0:
            print("Error: provide teams in pairs.", file=sys.stderr)
            sys.exit(1)
        for i in range(0, len(args.teams), 2):
            matchups.append((args.teams[i], args.teams[i + 1]))
    else:
        parser.print_help()
        return

    if not matchups:
        print("No matchups to predict.")
        return

    neutral = not args.not_neutral
    results = []
    for home, away in matchups:
        try:
            r = predict_match(
                home, away, neutral, args.knockout,
                model, cfg, final_ratings, df_full,
                form_data=form_data,
                form_data_v2=form_data_v2,
                form_alpha=args.form_alpha,
            )
            results.append(r)
        except ValueError as e:
            print(f"  Error: {e}", file=sys.stderr)

    print_results(results)


if __name__ == "__main__":
    main()
