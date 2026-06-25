"""
predict_j3.py — Predicciones jornada 3 WC 2026 (24-27 junio).

Genera predicciones para los 24 partidos de la jornada 3 antes de que
se jueguen. Guarda en predictions_j3.csv con columnas actual_home /
actual_away / correct para rellenar conforme se vayan jugando.

Run:
    py predict_j3.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.data.load import load_results
from src.features.elo import compute_elo
from src.features.h2h import compute_h2h
from src.features.tournament import add_tournament_features

cfg   = json.load(open("models/v5_config.json"))
model = joblib.load("models/poisson_dc_v5.joblib")
p, hp = cfg["priors"], cfg["hyperparams"]
THETA = cfg["theta_D"]

# Build pipeline state (Elo + H2H through J2 results)
df_raw = load_results()
df, final_ratings = compute_elo(df_raw, home_advantage=cfg["home_advantage"])
df = compute_h2h(df, p["global_home_avg"], p["global_away_avg"], k=hp["h2h_k"])
df = add_tournament_features(df)

# J3 fixtures from CSV (NA scores)
raw_csv = pd.read_csv("international_results-master/results.csv")
raw_csv["date"] = pd.to_datetime(raw_csv["date"])
j3 = raw_csv[
    (raw_csv["date"] >= "2026-06-24") &
    (raw_csv["date"] <= "2026-06-27") &
    (raw_csv["tournament"] == "FIFA World Cup")
].sort_values("date").reset_index(drop=True)

rows = []
for _, fix in j3.iterrows():
    home, away = fix["home_team"], fix["away_team"]
    r_h = final_ratings.get(home, 1500.0)
    r_a = final_ratings.get(away, 1500.0)
    elo_diff = r_h - r_a

    mask  = (df["home_team"] == home) & (df["away_team"] == away)
    hist  = df[mask]
    n_h2h = len(hist)
    if n_h2h == 0:
        h2h_home = p["global_home_avg"]
        h2h_away = p["global_away_avg"]
    else:
        h2h_home = (hist["home_score"].sum() + hp["h2h_k"] * p["global_home_avg"]) / (n_h2h + hp["h2h_k"])
        h2h_away = (hist["away_score"].sum() + hp["h2h_k"] * p["global_away_avg"]) / (n_h2h + hp["h2h_k"])

    feat = pd.DataFrame([{
        "elo_diff":          elo_diff,
        "neutral":           1.0,
        "h2h_home_goals_mu": h2h_home,
        "h2h_away_goals_mu": h2h_away,
        "is_world_cup":      1.0,
        "is_knockout":       0.0,
    }])

    proba     = model.predict_proba(feat)
    scoreline = model.predict_scoreline(feat)
    p_h, p_d, p_a = float(proba[0, 0]), float(proba[0, 1]), float(proba[0, 2])
    pred_h = int(scoreline["pred_home"].iloc[0])
    pred_a = int(scoreline["pred_away"].iloc[0])

    if p_d > THETA:  pred = "D"
    elif p_h >= p_a: pred = home
    else:            pred = away

    rows.append({
        "date":        str(fix["date"])[:10],
        "home_team":   home,
        "away_team":   away,
        "home_elo":    round(r_h),
        "away_elo":    round(r_a),
        "elo_diff":    round(elo_diff),
        "h2h_n":       n_h2h,
        "P_home":      round(p_h, 3),
        "P_draw":      round(p_d, 3),
        "P_away":      round(p_a, 3),
        "pred_score":  f"{pred_h}-{pred_a}",
        "prediction":  pred,
        "actual_home": "",
        "actual_away": "",
        "correct":     "",
    })

out = pd.DataFrame(rows)

# ── print ──────────────────────────────────────────────────────────────────────
print(f"Jornada 3 WC 2026 — predicciones del modelo v5 (theta_D={THETA})\n")
print(f"  {'Fecha':<12} {'Local':<26} {'Visitante':<26} {'P(L)':>5} {'P(E)':>5} {'P(V)':>5}  {'Score':>5}  Pred")
print("  " + "-" * 102)
for _, r in out.iterrows():
    h_lbl = r["home_team"] + ("*" if r["prediction"] == r["home_team"] else "")
    a_lbl = r["away_team"] + ("*" if r["prediction"] == r["away_team"] else "")
    pred_lbl = "Empate" if r["prediction"] == "D" else r["prediction"]
    print(
        f"  {r['date']:<12} {h_lbl:<26} {a_lbl:<26}"
        f"  {r['P_home']:>5.3f} {r['P_draw']:>5.3f} {r['P_away']:>5.3f}"
        f"  {r['pred_score']:>5}  {pred_lbl}"
    )

n_home = (out["prediction"] == out["home_team"]).sum()
n_away = (out["prediction"] == out["away_team"]).sum()
n_draw = (out["prediction"] == "D").sum()
print(f"\n  Resumen: {n_home} victorias local / {n_draw} empates / {n_away} victorias visitante")
print(f"  Modelo Elo favorito en todos — empates solo si P(D) > {THETA}")

# ── save ───────────────────────────────────────────────────────────────────────
out_path = Path("predictions_j3.csv")
out.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\nGuardado: {out_path}")
print("  Rellena actual_home / actual_away conforme se vayan jugando.")
print("  Columna 'correct' se puede calcular post-hoc comparando prediction vs resultado real.")
