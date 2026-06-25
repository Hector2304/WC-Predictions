"""
Validates v5 vs form_v2 vs form_v3 on completed WC 2026 matches.

Leakage-free: for each match, form is computed using only matches
played strictly before that date. FIFA data is a static snapshot
(mild leakage for early matches — noted in output).

Run from project root:
    py validate_experimental.py
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, brier_score_loss

warnings.filterwarnings("ignore")

from src.data.load import load_results
from src.features.elo import compute_elo
from src.features.h2h import compute_h2h
from src.features.tournament import add_tournament_features
from src.features.tournament_form import (
    get_wc_form_v2,
    form_adjusted_lambdas_v2,
    form_adjusted_lambdas_v3,
)
from src.features.fifa_stats import load_latest_fifa_data

FIFA_DATA_DIR = "DATOS FIFA"
H2H_K        = 5.0
CLASSES      = ["H", "D", "A"]


def label(hs, as_):
    if hs > as_: return "H"
    if hs < as_: return "A"
    return "D"


def apply_threshold(ph, pd_, pa, theta):
    if pd_ > theta: return "D"
    return "H" if ph >= pa else "A"


def build_row(home, away, ratings, df_full, cfg):
    p = cfg["priors"]
    k = H2H_K
    r_h = ratings.get(home, 1500)
    r_a = ratings.get(away, 1500)

    mask = (df_full["home_team"] == home) & (df_full["away_team"] == away)
    hist = df_full[mask]
    n    = len(hist)
    h2h_h = (hist["home_score"].sum() + k * p["global_home_avg"]) / (n + k) if n else p["global_home_avg"]
    h2h_a = (hist["away_score"].sum() + k * p["global_away_avg"]) / (n + k) if n else p["global_away_avg"]

    return pd.DataFrame([{
        "elo_diff":          r_h - r_a,
        "neutral":           1.0,
        "h2h_home_goals_mu": h2h_h,
        "h2h_away_goals_mu": h2h_a,
        "is_world_cup":      1.0,
        "is_knockout":       0.0,
    }])


# ── load pipeline ─────────────────────────────────────────────────────────────
with open("models/v5_config.json") as f:
    cfg = json.load(f)

model    = joblib.load("models/poisson_dc_v5.joblib")
df_raw   = load_results()
df_full, ratings = compute_elo(df_raw)
df_full  = compute_h2h(df_full, cfg["priors"]["global_home_avg"],
                        cfg["priors"]["global_away_avg"], k=H2H_K)
df_full  = add_tournament_features(df_full)
fifa     = load_latest_fifa_data(FIFA_DATA_DIR)

theta    = cfg["theta_D"]

# ── WC 2026 completed matches ─────────────────────────────────────────────────
wc = df_full[
    (df_full["date"].dt.year == 2026)
    & (df_full["tournament"].str.lower() == "fifa world cup")
    & df_full["home_score"].notna()
].sort_values("date").copy()

print(f"Partidos WC 2026 con resultado: {len(wc)}")
print(f"FIFA data: {'cargada (' + str(len(fifa)) + ' equipos)' if fifa else 'NO disponible'}")
print(f"Nota: FIFA data es un snapshot estático — leve leakage en partidos tempranos\n")

# ── predict each match ────────────────────────────────────────────────────────
rows = []

for _, match in wc.iterrows():
    home   = match["home_team"]
    away   = match["away_team"]
    actual = label(int(match["home_score"]), int(match["away_score"]))
    date   = match["date"]

    # Form computed from matches STRICTLY before this date (leakage-free)
    df_prior = df_full[df_full["date"] < date]
    wc_form_v2, avg_qa_gpg, avg_qa_cpg = get_wc_form_v2(df_prior, ratings)

    row = build_row(home, away, ratings, df_full, cfg)

    # v5
    lam_h_arr, lam_a_arr = model.predict_lambdas(row)
    lam_h = float(lam_h_arr[0])
    lam_a = float(lam_a_arr[0])
    proba = model.predict_proba(row)
    ph_v5, pd_v5, pa_v5 = float(proba[0,0]), float(proba[0,1]), float(proba[0,2])
    pred_v5 = apply_threshold(ph_v5, pd_v5, pa_v5, theta)

    # form_v2
    lh2, la2 = form_adjusted_lambdas_v2(lam_h, lam_a, home, away,
                                         wc_form_v2, avg_qa_gpg, avg_qa_cpg)
    pr2, _, _ = model.predict_from_lambdas(np.array([lh2]), np.array([la2]))
    ph_v2, pd_v2, pa_v2 = float(pr2[0,0]), float(pr2[0,1]), float(pr2[0,2])
    pred_v2 = apply_threshold(ph_v2, pd_v2, pa_v2, theta)

    # form_v3
    if fifa:
        lh3, la3 = form_adjusted_lambdas_v3(lam_h, lam_a, home, away,
                                              wc_form_v2, avg_qa_gpg, avg_qa_cpg, fifa)
        pr3, _, _ = model.predict_from_lambdas(np.array([lh3]), np.array([la3]))
        ph_v3, pd_v3, pa_v3 = float(pr3[0,0]), float(pr3[0,1]), float(pr3[0,2])
        pred_v3 = apply_threshold(ph_v3, pd_v3, pa_v3, theta)
    else:
        ph_v3, pd_v3, pa_v3 = ph_v2, pd_v2, pa_v2
        pred_v3 = pred_v2

    rows.append({
        "date":    date.strftime("%m-%d"),
        "match":   f"{home[:12]} vs {away[:12]}",
        "actual":  actual,
        "v5":      pred_v5,
        "v2":      pred_v2,
        "v3":      pred_v3,
        "ok_v5":   pred_v5 == actual,
        "ok_v2":   pred_v2 == actual,
        "ok_v3":   pred_v3 == actual,
        # probas for log-loss / brier
        "ph_v5": ph_v5, "pd_v5": pd_v5, "pa_v5": pa_v5,
        "ph_v2": ph_v2, "pd_v2": pd_v2, "pa_v2": pa_v2,
        "ph_v3": ph_v3, "pd_v3": pd_v3, "pa_v3": pa_v3,
        # where models differ
        "v2_vs_v5": pred_v2 != pred_v5,
        "v3_vs_v2": pred_v3 != pred_v2,
    })

df_r = pd.DataFrame(rows)

# ── summary metrics ───────────────────────────────────────────────────────────
actual_labels = df_r["actual"].tolist()

def proba_matrix(df, prefix):
    return df[[f"ph_{prefix}", f"pd_{prefix}", f"pa_{prefix}"]].values

ll_v5 = log_loss(actual_labels, proba_matrix(df_r, "v5"), labels=["H","D","A"])
ll_v2 = log_loss(actual_labels, proba_matrix(df_r, "v2"), labels=["H","D","A"])
ll_v3 = log_loss(actual_labels, proba_matrix(df_r, "v3"), labels=["H","D","A"])

f1_v5 = f1_score(actual_labels, df_r["v5"], labels=CLASSES, average="macro", zero_division=0)
f1_v2 = f1_score(actual_labels, df_r["v2"], labels=CLASSES, average="macro", zero_division=0)
f1_v3 = f1_score(actual_labels, df_r["v3"], labels=CLASSES, average="macro", zero_division=0)

acc_v5 = df_r["ok_v5"].mean()
acc_v2 = df_r["ok_v2"].mean()
acc_v3 = df_r["ok_v3"].mean()

def multiclass_brier(actual_labels, pm, classes=CLASSES):
    Y = np.zeros((len(actual_labels), len(classes)))
    for i, a in enumerate(actual_labels):
        Y[i, classes.index(a)] = 1.0
    return float(np.mean(np.sum((pm - Y) ** 2, axis=1)))

bs_v5 = multiclass_brier(actual_labels, proba_matrix(df_r, "v5"))
bs_v2 = multiclass_brier(actual_labels, proba_matrix(df_r, "v2"))
bs_v3 = multiclass_brier(actual_labels, proba_matrix(df_r, "v3"))

n = len(df_r)
print("=" * 66)
print(f"{'METRICAS':32s}  {'v5':>8}  {'form_v2':>8}  {'form_v3':>8}")
print("-" * 66)
print(f"{'Accuracy':32s}  {acc_v5:8.1%}  {acc_v2:8.1%}  {acc_v3:8.1%}")
print(f"{'F1 macro':32s}  {f1_v5:8.4f}  {f1_v2:8.4f}  {f1_v3:8.4f}")
print(f"{'Brier multiclase (lower=better)':32s}  {bs_v5:8.4f}  {bs_v2:8.4f}  {bs_v3:8.4f}")
print(f"{'Log-loss   (lower=better)':32s}  {ll_v5:8.4f}  {ll_v2:8.4f}  {ll_v3:8.4f}")
print(f"{'Correctos / Total':32s}  {df_r['ok_v5'].sum():>4}/{n}    {df_r['ok_v2'].sum():>4}/{n}    {df_r['ok_v3'].sum():>4}/{n}")
print("=" * 66)

# ── veredicto automatico por metrica ─────────────────────────────────────────
def winner(a, b, lower_is_better=False):
    if lower_is_better:
        return "form_v2" if b < a else ("v5" if a < b else "empate")
    return "form_v2" if b > a else ("v5" if a > b else "empate")

print(f"\n  Accuracy  -> gana {winner(acc_v5, acc_v2)}")
print(f"  Brier     -> gana {winner(bs_v5,  bs_v2,  lower_is_better=True)}")
print(f"  Log-loss  -> gana {winner(ll_v5,  ll_v2,  lower_is_better=True)}")
print(f"\n  Interpretacion:")
print(f"  - Si Brier mejora pero accuracy baja: form_v2 esta mejor calibrado")
print(f"    (probabilidades mas honestas, no necesariamente mas predicciones correctas)")
print(f"  - Para bracket simulation: usar el modelo con menor Brier")
print(f"  - Para predecir partido a partido: usar el modelo con mayor accuracy")

# ── diagnostico double-push: form_v2 + theta_D ───────────────────────────────
diff_v2 = df_r[df_r["v2_vs_v5"]]
diff_v3 = df_r[df_r["v3_vs_v2"]]

print(f"\nform_v2 cambio la prediccion de v5 en {len(diff_v2)}/{n} partidos")
print(f"form_v3 cambio la prediccion de v2  en {len(diff_v3)}/{n} partidos")

print(f"\nDIAGNOSTICO DOUBLE-PUSH (theta_D={theta:.2f})")
print(f"  Partidos donde form_v2 cruzo el threshold de empate:")
print(f"  {'Partido':<28} {'Real':>4}  {'P(D)v5':>7}  {'P(D)v2':>7}  {'Shift':>6}  {'Pred':>4}  {'ok':>4}")
print("  " + "-" * 66)

# All matches where P(Draw) crossed theta_D (in either direction)
for _, r in df_r.iterrows():
    v5_over  = r["pd_v5"] > theta
    v2_over  = r["pd_v2"] > theta
    if v5_over != v2_over:   # threshold crossed by form adjustment
        shift = r["pd_v2"] - r["pd_v5"]
        direction = "v5->D" if v2_over else "D->v5"
        ok = "OK" if r["ok_v2"] else "--"
        print(f"  {r['match']:<28} {r['actual']:>4}  {r['pd_v5']:7.3f}  {r['pd_v2']:7.3f}  {shift:+6.3f}  {r['v2']:>4}  {ok:>4}  [{direction}]")

# Wider view: avg P(Draw) shift across all matches
avg_shift = (df_r["pd_v2"] - df_r["pd_v5"]).mean()
pct_over_v5 = (df_r["pd_v5"] > theta).mean()
pct_over_v2 = (df_r["pd_v2"] > theta).mean()
print(f"\n  Shift promedio en P(Draw):  {avg_shift:+.4f}")
print(f"  % partidos sobre theta_D:  v5={pct_over_v5:.1%}  form_v2={pct_over_v2:.1%}")
print(f"  -> {'form_v2 empuja hacia D sistematicamente (double-push activo)' if avg_shift > 0.005 else 'No hay sesgo sistematico hacia D'}")

# ── v3 decision rule ─────────────────────────────────────────────────────────
print(f"\nREGLA DE DECISION PARA v3 (evaluar en eliminatoria):")
v3_changed = int(df_r["v3_vs_v2"].sum())
if v3_changed == 0:
    print(f"  v3 cambio 0/{n} predicciones -> INERTE en grupos")
    print(f"  En eliminatoria (n esperado ~16-32 partidos):")
    print(f"    0 cambios -> subir alpha_fifa de 0.90 a 0.80, o eliminar capa")
    print(f"    1+ cambios con Brier mejor -> mantener o subir peso")
    print(f"    1+ cambios con Brier peor  -> bajar peso o eliminar")
else:
    v3_ok  = df_r[df_r["v3_vs_v2"] & df_r["ok_v3"]].shape[0]
    v3_bad = v3_changed - v3_ok
    print(f"  v3 cambio {v3_changed}/{n}: {v3_ok} correctos, {v3_bad} incorrectos")
    print(f"  Brier v2={bs_v2:.4f} vs v3={bs_v3:.4f} -> {'MANTENER v3' if bs_v3 < bs_v2 else 'REVISAR peso FIFA'}")

if len(diff_v3) > 0:
    print(f"\nPartidos donde v3 difiere de v2:")
    print(f"  {'Partido':<28} {'Real':>4}  {'v5':>4}  {'v2':>4}  {'v3':>4}  {'v3 ok?':>6}")
    print("  " + "-" * 55)
    for _, r in diff_v3.iterrows():
        ok = "OK" if r["ok_v3"] else "--"
        was_ok_v2 = "OK" if r["ok_v2"] else "--"
        print(f"  {r['match']:<28} {r['actual']:>4}  {r['v5']:>4}  {r['v2']:>4}  {r['v3']:>4}  {ok:>6}  (v2 era {was_ok_v2})")

# ── full match table ───────────────────────────────────────────────────────────
print("\n\nTABLA COMPLETA")
print(f"  {'Fecha':>5}  {'Partido':<28} {'Real':>4}  {'v5':>4}  {'v2':>4}  {'v3':>4}  {'v5':>4}  {'v2':>4}  {'v3':>4}")
print(f"  {'':>5}  {'':28} {'':>4}  {'':>4}  {'':>4}  {'':>4}  {'ok':>4}  {'ok':>4}  {'ok':>4}")
print("  " + "-" * 72)
for _, r in df_r.iterrows():
    tag = " ←" if r["v3_vs_v2"] else ""
    print(
        f"  {r['date']:>5}  {r['match']:<28} {r['actual']:>4}  "
        f"{r['v5']:>4}  {r['v2']:>4}  {r['v3']:>4}  "
        f"{'OK' if r['ok_v5'] else '--':>4}  "
        f"{'OK' if r['ok_v2'] else '--':>4}  "
        f"{'OK' if r['ok_v3'] else '--':>4}{tag}"
    )

# ── result distribution ────────────────────────────────────────────────────────
print("\n\nDISTRIBUCIÓN DE PREDICCIONES vs REAL")
for model_name, col in [("Real", "actual"), ("v5", "v5"), ("form_v2", "v2"), ("form_v3", "v3")]:
    counts = df_r[col].value_counts().to_dict()
    h = counts.get("H", 0)
    d = counts.get("D", 0)
    a = counts.get("A", 0)
    print(f"  {model_name:<10}  H={h:>2}  D={d:>2}  A={a:>2}")
