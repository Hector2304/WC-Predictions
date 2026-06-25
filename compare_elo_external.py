"""
compare_elo_external.py — Compara nuestro Elo vs fuente externa (24 jun 2026).

Filtra solo equipos del WC 2026, compara rankings (no valores absolutos —
las escalas son distintas), e identifica discrepancias grandes.

Run:
    py compare_elo_external.py
"""

import json
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.data.load import load_results
from src.features.elo import compute_elo

cfg = json.load(open("models/v5_config.json"))
df_raw = load_results()
_, final_ratings = compute_elo(df_raw, home_advantage=cfg["home_advantage"])

# External Elo — nombres mapeados al inglés del dataset
external = {
    "Argentina": 1901.93, "France": 1894.40, "Spain": 1864.32,
    "England": 1829.82, "Brazil": 1772.01, "Morocco": 1769.98,
    "Portugal": 1766.74, "Netherlands": 1764.40, "Germany": 1760.46,
    "Belgium": 1727.88, "Colombia": 1727.42, "Mexico": 1721.78,
    "Croatia": 1711.48, "United States": 1709.59, "Japan": 1681.26,
    "Switzerland": 1654.94, "Uruguay": 1649.96, "Iran": 1611.21,
    "Norway": 1606.48, "Austria": 1599.99, "South Korea": 1591.75,
    "Australia": 1584.55, "Egypt": 1583.37, "Algeria": 1575.64,
    "Canada": 1572.13, "Ecuador": 1558.35, "Ivory Coast": 1551.71,
    "Turkey": 1550.13, "Sweden": 1517.99, "Paraguay": 1517.39,
    "Scotland": 1504.41, "Panama": 1489.05, "Czech Republic": 1481.49,
    "DR Congo": 1472.37, "Qatar": 1438.82, "Tunisia": 1437.69,
    "Uzbekistan": 1432.84, "Saudi Arabia": 1426.71, "Iraq": 1419.24,
    "South Africa": 1418.21, "Cape Verde": 1401.77, "Ghana": 1398.57,
    "Bosnia and Herzegovina": 1381.18, "Jordan": 1355.89,
    "Senegal": 1638.36, "New Zealand": 1277.34, "Haiti": 1271.00,
    "Curaçao": 1299.41,
}

# Check coverage
wc_teams = set(external.keys())
missing = wc_teams - set(final_ratings.keys())
if missing:
    print(f"AVISO — equipos en fuente externa no encontrados en nuestro modelo: {missing}\n")

our_wc = {t: final_ratings[t] for t in wc_teams if t in final_ratings}

# Rank within WC teams only
our_df = (pd.DataFrame({"team": list(our_wc.keys()), "elo_ours": list(our_wc.values())})
          .sort_values("elo_ours", ascending=False).reset_index(drop=True))
our_df["rank_ours"] = our_df.index + 1

ext_df = (pd.DataFrame({"team": list(external.keys()), "elo_ext": list(external.values())})
          .sort_values("elo_ext", ascending=False).reset_index(drop=True))
ext_df["rank_ext"] = ext_df.index + 1

df = our_df.merge(ext_df, on="team")
df["rank_diff"] = df["rank_ours"] - df["rank_ext"]
df = df.sort_values("rank_ext").reset_index(drop=True)

# ── tabla completa ────────────────────────────────────────────────────────────
print("Comparacion Elo — WC 2026 (48 equipos, ordenado por ranking externo)")
print("rank_diff: negativo = nuestro modelo lo pone MAS ALTO que el externo")
print()
print(f"  {'Equipo':<28} {'R.ext':>5} {'R.nuestro':>9} {'Dif':>5}  {'Elo ext':>8}  {'Elo nuestro':>11}")
print("  " + "-" * 75)
for _, r in df.iterrows():
    diff = int(r["rank_diff"])
    flag = "  ***" if abs(diff) >= 8 else ("  *" if abs(diff) >= 4 else "")
    sign = f"+{diff}" if diff > 0 else str(diff)
    print(f"  {r['team']:<28} {int(r['rank_ext']):>5} {int(r['rank_ours']):>9} {sign:>5}"
          f"  {r['elo_ext']:>8.1f}  {r['elo_ours']:>11.0f}{flag}")

# ── discrepancias grandes ────────────────────────────────────────────────────
print()
print("=" * 60)
print("  Discrepancias grandes (|rank_diff| >= 4)")
print("=" * 60)
big = df[df["rank_diff"].abs() >= 4].sort_values("rank_diff")
print(f"\n  Equipos que nuestro modelo pone DEMASIADO ALTO (rank_diff negativo):")
for _, r in big[big["rank_diff"] < 0].iterrows():
    print(f"    {r['team']:<28}  ext=#{int(r['rank_ext'])}  nuestro=#{int(r['rank_ours'])}  diff={int(r['rank_diff'])}")
print(f"\n  Equipos que nuestro modelo pone DEMASIADO BAJO (rank_diff positivo):")
for _, r in big[big["rank_diff"] > 0].iterrows():
    print(f"    {r['team']:<28}  ext=#{int(r['rank_ext'])}  nuestro=#{int(r['rank_ours'])}  diff=+{int(r['rank_diff'])}")

# ── metricas globales ─────────────────────────────────────────────────────────
from scipy.stats import spearmanr
rho, pval = spearmanr(df["rank_ours"], df["rank_ext"])
mae = df["rank_diff"].abs().mean()
print()
print("=" * 60)
print(f"  Correlacion Spearman de rankings: {rho:.4f}  (p={pval:.4f})")
print(f"  MAE rankings:                     {mae:.1f} posiciones promedio")
print(f"  Max discrepancia:                 {df['rank_diff'].abs().max():.0f} posiciones")
print()
if rho > 0.90:
    print("  VEREDICTO: rankings muy correlacionados. Diferencias son ruido de escala")
    print("  o partidos no FIFA que inflan/deflacionan algunos equipos. No reentrenar.")
elif rho > 0.80:
    print("  VEREDICTO: correlacion buena pero hay discrepancias sistematicas.")
    print("  Revisar equipos con diff grande antes de decidir.")
else:
    print("  VEREDICTO: correlacion baja. Las diferencias son sustanciales.")
    print("  Vale la pena investigar la causa antes del R16.")
