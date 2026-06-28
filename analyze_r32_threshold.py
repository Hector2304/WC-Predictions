"""
Analiza el R32 2026 con theta_D=0.26 vs 0.28 para decidir si se necesita
un threshold propio para esta ronda.

Run: py analyze_r32_threshold.py
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.data.load import load_results
from src.features.elo import compute_elo
from src.features.h2h import compute_h2h
from src.features.tournament import add_tournament_features

FIXTURE = "international_results-master/r32_2026.csv"
CLASSES = ["H", "D", "A"]
T26     = 0.26
T28     = 0.28


def apply_threshold(ph, pd_, pa, theta):
    if pd_ > theta: return "D"
    return "H" if ph >= pa else "A"


with open("models/v5_config.json") as f:
    cfg = json.load(f)

model = joblib.load("models/poisson_dc_v5.joblib")
p     = cfg["priors"]
k     = cfg["hyperparams"]["h2h_k"]

df_raw = load_results()
df, ratings = compute_elo(df_raw, home_advantage=cfg.get("home_advantage", 20.0))
df = compute_h2h(df, p["global_home_avg"], p["global_away_avg"], k=k)
df = add_tournament_features(df)

fixture = pd.read_csv(FIXTURE)

rows = []
for _, m in fixture.iterrows():
    home = m["home"]
    away = m["away"]
    r_h  = ratings.get(home, 1500)
    r_a  = ratings.get(away, 1500)

    # normalizar: mayor Elo como "home" en KO neutral
    swapped = False
    if r_a > r_h:
        home, away = away, home
        r_h, r_a   = r_a, r_h
        swapped = True

    elo_diff = r_h - r_a

    mask     = (df["home_team"] == home) & (df["away_team"] == away)
    h2h_hist = df[mask]
    n        = len(h2h_hist)
    h2h_h    = (h2h_hist["home_score"].sum() + k * p["global_home_avg"]) / (n + k) if n else p["global_home_avg"]
    h2h_a    = (h2h_hist["away_score"].sum() + k * p["global_away_avg"]) / (n + k) if n else p["global_away_avg"]

    row = pd.DataFrame([{
        "elo_diff":          elo_diff,
        "neutral":           1.0,
        "h2h_home_goals_mu": h2h_h,
        "h2h_away_goals_mu": h2h_a,
        "is_world_cup":      1.0,
        "is_knockout":       1.0,
    }])

    proba = model.predict_proba(row)
    ph, pd_, pa = float(proba[0,0]), float(proba[0,1]), float(proba[0,2])

    orig_home = m["home"]
    orig_away = m["away"]
    fav  = home  # mayor Elo
    und  = away

    pred26 = apply_threshold(ph, pd_, pa, T26)
    pred28 = apply_threshold(ph, pd_, pa, T28)

    # traducir pred de vuelta al fixture original
    def label_to_orig(pred, swapped):
        if pred == "D": return "D"
        if not swapped:
            return orig_home if pred == "H" else orig_away
        else:
            return orig_home if pred == "A" else orig_away

    rows.append({
        "partido":   f"{orig_home} vs {orig_away}",
        "fecha":     m["date"],
        "elo_fav":   fav,
        "elo_diff":  elo_diff,
        "P(H)":      ph,
        "P(D)":      pd_,
        "P(A)":      pa,
        "pred_0.26": label_to_orig(pred26, swapped),
        "pred_0.28": label_to_orig(pred28, swapped),
        "cambia":    pred26 != pred28,
        "zona_gris": abs(pd_ - T28) < 0.03,
    })

df_r = pd.DataFrame(rows)

print("=" * 80)
print("ANÁLISIS R32 2026 — theta_D=0.26 vs 0.28")
print("=" * 80)
print(f"\n{'Partido':<34} {'Fecha':>10}  {'ELO diff':>8}  {'P(D)':>6}  {'t=0.26':>10}  {'t=0.28':>10}  {'¿Cambia?'}")
print("-" * 100)
for _, r in df_r.iterrows():
    marca = " ** CAMBIA" if r["cambia"] else ("  ~gris" if r["zona_gris"] else "")
    print(
        f"{r['partido']:<34} {r['fecha']:>10}  {r['elo_diff']:>8.0f}  "
        f"{r['P(D)']:>6.3f}  {str(r['pred_0.26']):>10}  {str(r['pred_0.28']):>10}{marca}"
    )

n_cambia   = df_r["cambia"].sum()
n_gris     = df_r["zona_gris"].sum()
avg_pd     = df_r["P(D)"].mean()
over28     = (df_r["P(D)"] > T28).sum()
over26     = (df_r["P(D)"] > T26).sum()

print("-" * 100)
print(f"\nRESUMEN")
print(f"  Partidos que cambian de predicción entre t=0.26 y t=0.28 : {n_cambia}/{len(df_r)}")
print(f"  Partidos en zona gris (P(D) entre 0.25 y 0.31)           : {n_gris}/{len(df_r)}")
print(f"  P(D) promedio del R32                                     : {avg_pd:.3f}")
print(f"  Partidos con P(D) > 0.26                                  : {over26}/{len(df_r)}")
print(f"  Partidos con P(D) > 0.28                                  : {over28}/{len(df_r)}")
print(f"\n  Referencia histórica: R16=19.6% empates, QF=32.1%, SF=14.3%")
print(f"  P(D) promedio actual sugiere tasa esperada ~= {avg_pd:.1%}")

if n_cambia == 0:
    print(f"\n  CONCLUSIÓN: Los dos thresholds dan las mismas 16 predicciones.")
    print(f"  El threshold no es relevante para el R32 con este fixture.")
    print(f"  Recomendación: usar theta_D_knockout=0.28 (valor actual, sin cambio).")
elif n_cambia <= 2:
    print(f"\n  CONCLUSIÓN: Solo {n_cambia} partido(s) cambian.")
    print(f"  Revisar esos casos manualmente para decidir.")
else:
    print(f"\n  CONCLUSIÓN: {n_cambia} partidos cambian — el threshold sí importa para el R32.")
    print(f"  Considerar theta_D_R32=0.26 separado de theta_D_knockout=0.28.")
