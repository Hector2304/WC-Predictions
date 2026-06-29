"""
Genera predictions_r32.csv con columnas para v5 puro y experimental (form_v3).

Uso:
    py predict_r32.py
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson as _pois

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.data.load import load_results
from src.features.elo import compute_elo
from src.features.h2h import compute_h2h
from src.features.tournament import add_tournament_features
from src.features.tournament_form import (
    form_adjusted_lambdas_v3,
    get_wc_form,
    get_wc_form_v2,
)
from src.features.fifa_stats import load_latest_fifa_data

FIFA_DATA_DIR = "DATOS FIFA"
R32_CSV      = "international_results-master/r32_2026.csv"
OUT_CSV      = "predictions_r32.csv"


def load_pipeline():
    with open("models/v5_config.json") as f:
        cfg = json.load(f)
    model = joblib.load("models/poisson_dc_v5.joblib")

    ha = cfg.get("home_advantage", 20.0)
    p  = cfg["priors"]
    hp = cfg["hyperparams"]

    df_raw = load_results()
    df, final_ratings = compute_elo(df_raw, home_advantage=ha)
    df = compute_h2h(df, p["global_home_avg"], p["global_away_avg"], k=hp["h2h_k"])
    df = add_tournament_features(df)

    wc_form, wc_avg_gpg                        = get_wc_form(df)
    wc_form_v2, wc_avg_qa_gpg, wc_avg_qa_cpg  = get_wc_form_v2(df, final_ratings)
    fifa_data = load_latest_fifa_data(FIFA_DATA_DIR)

    return model, cfg, df, final_ratings, wc_form, wc_avg_gpg, wc_form_v2, wc_avg_qa_gpg, wc_avg_qa_cpg, fifa_data


def predict_match(home, away, neutral, knockout,
                  model, cfg, df_full, final_ratings,
                  wc_form_v2, wc_avg_qa_gpg, wc_avg_qa_cpg,
                  fifa_data, form_alpha=0.7):
    p  = cfg["priors"]
    k  = cfg["hyperparams"]["h2h_k"]
    theta_D = cfg["theta_D_knockout"] if knockout else cfg["theta_D"]

    r_h = final_ratings.get(home, 1500.0)
    r_a = final_ratings.get(away, 1500.0)

    # Swap to higher-Elo "home" for deterministic H2H (same logic as app.py)
    swapped = False
    h, a = home, away
    if knockout and neutral and r_a > r_h:
        h, a = a, h
        r_h, r_a = r_a, r_h
        swapped = True

    mask  = (df_full["home_team"] == h) & (df_full["away_team"] == a)
    hist  = df_full[mask]
    n_h2h = len(hist)

    if n_h2h == 0:
        h2h_home = p["global_home_avg"]
        h2h_away = p["global_away_avg"]
    else:
        h2h_home = (hist["home_score"].sum() + k * p["global_home_avg"]) / (n_h2h + k)
        h2h_away = (hist["away_score"].sum() + k * p["global_away_avg"]) / (n_h2h + k)

    row = pd.DataFrame([{
        "elo_diff":          r_h - r_a,
        "neutral":           float(neutral),
        "h2h_home_goals_mu": h2h_home,
        "h2h_away_goals_mu": h2h_away,
        "is_world_cup":      1.0,
        "is_knockout":       1.0 if knockout else 0.0,
    }])

    lam_h_arr, lam_a_arr = model.predict_lambdas(row)
    lam_h = float(lam_h_arr[0])
    lam_a = float(lam_a_arr[0])

    proba     = model.predict_proba(row)
    scoreline = model.predict_scoreline(row)

    p_h = float(proba[0, 0])
    p_d = float(proba[0, 1])
    p_a = float(proba[0, 2])
    sh  = int(scoreline["pred_home"].iloc[0])
    sa  = int(scoreline["pred_away"].iloc[0])

    # Unswap v5 probs/score to original fixture order
    if swapped:
        p_h, p_a = p_a, p_h
        sh, sa   = sa, sh

    if p_d > theta_D:
        v5_pred = "D"
    elif p_h >= p_a:
        v5_pred = home
    else:
        v5_pred = away

    # ── form_v3: form_v2 + FIFA stats ────────────────────────────────────────
    lam_h_v3, lam_a_v3 = form_adjusted_lambdas_v3(
        lam_h, lam_a,
        h, a,
        wc_form_v2, wc_avg_qa_gpg, wc_avg_qa_cpg,
        fifa_data,
        alpha_form=form_alpha,
        alpha_fifa=0.90,
    )
    proba_v3, sh_v3_arr, sa_v3_arr = model.predict_from_lambdas(
        np.array([lam_h_v3]), np.array([lam_a_v3])
    )
    ph_v3 = float(proba_v3[0, 0])
    pd_v3 = float(proba_v3[0, 1])
    pa_v3 = float(proba_v3[0, 2])
    sh_v3 = int(sh_v3_arr[0])
    sa_v3 = int(sa_v3_arr[0])

    # Unswap form_v3
    if swapped:
        ph_v3, pa_v3 = pa_v3, ph_v3
        sh_v3, sa_v3 = sa_v3, sh_v3

    if pd_v3 > theta_D:
        exp_pred = "D"
    elif ph_v3 >= pa_v3:
        exp_pred = home
    else:
        exp_pred = away

    # elo_diff from original fixture perspective
    elo_h = final_ratings.get(home, 1500.0)
    elo_a = final_ratings.get(away, 1500.0)

    return {
        "home_elo":   round(elo_h),
        "away_elo":   round(elo_a),
        "elo_diff":   round(elo_h - elo_a),
        "h2h_n":      n_h2h,
        "v5_P_home":  round(p_h, 3),
        "v5_P_draw":  round(p_d, 3),
        "v5_P_away":  round(p_a, 3),
        "v5_score":   f"{sh}-{sa}",
        "v5_pred":    v5_pred,
        "exp_P_home": round(ph_v3, 3),
        "exp_P_draw": round(pd_v3, 3),
        "exp_P_away": round(pa_v3, 3),
        "exp_score":  f"{sh_v3}-{sa_v3}",
        "exp_pred":   exp_pred,
    }


def main():
    print("Cargando pipeline...")
    (model, cfg, df_full, final_ratings,
     wc_form, wc_avg_gpg,
     wc_form_v2, wc_avg_qa_gpg, wc_avg_qa_cpg,
     fifa_data) = load_pipeline()

    fixtures = pd.read_csv(R32_CSV)
    rows = []

    for _, fix in fixtures.iterrows():
        home    = fix["home"]
        away    = fix["away"]
        neutral = bool(fix["neutral"])
        date    = fix["date"]

        r = predict_match(
            home, away, neutral, knockout=True,
            model=model, cfg=cfg, df_full=df_full, final_ratings=final_ratings,
            wc_form_v2=wc_form_v2, wc_avg_qa_gpg=wc_avg_qa_gpg,
            wc_avg_qa_cpg=wc_avg_qa_cpg, fifa_data=fifa_data,
        )

        rows.append({
            "date":       date,
            "home_team":  home,
            "away_team":  away,
            "home_elo":   r["home_elo"],
            "away_elo":   r["away_elo"],
            "elo_diff":   r["elo_diff"],
            "h2h_n":      r["h2h_n"],
            "v5_P_home":  r["v5_P_home"],
            "v5_P_draw":  r["v5_P_draw"],
            "v5_P_away":  r["v5_P_away"],
            "v5_score":   r["v5_score"],
            "v5_pred":    r["v5_pred"],
            "exp_P_home": r["exp_P_home"],
            "exp_P_draw": r["exp_P_draw"],
            "exp_P_away": r["exp_P_away"],
            "exp_score":  r["exp_score"],
            "exp_pred":   r["exp_pred"],
            "actual_home":  "",
            "actual_away":  "",
            "v5_correct":   "",
            "exp_correct":  "",
        })

        v5_changed  = r["v5_pred"]  != r["exp_pred"]
        change_mark = " <-- DIFIEREN" if v5_changed else ""
        print(f"  {home:<30} vs {away:<30}  v5={r['v5_pred']:<25}  exp={r['exp_pred']}{change_mark}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nGuardado: {OUT_CSV}  ({len(rows)} partidos)")

    # Resumen de diferencias
    diffs = [(r["home_team"], r["away_team"], r["v5_pred"], r["exp_pred"])
             for r in rows if r["v5_pred"] != r["exp_pred"]]
    if diffs:
        print(f"\n{len(diffs)} partido(s) donde v5 y experimental difieren:")
        for home, away, v5, exp in diffs:
            print(f"  {home} vs {away}: v5={v5}  exp={exp}")
    else:
        print("\nV5 y experimental coinciden en todos los partidos.")


if __name__ == "__main__":
    main()
