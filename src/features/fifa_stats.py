"""
FIFA in-tournament stats loader for form_v3 experimental layer.

Loads 5 CSV files from DATOS FIFA/ (or the latest phase subfolder)
and returns per-team attack / defense factors normalized to the
tournament average.

Attack pool  (higher → team scores more):
  xG, xG efficiency, possession %, runs behind defense,
  receptions under pressure.

Defense pool (higher → team concedes less):
  goalkeeper saves, clean sheets, defensive pressures,
  ball recovery time (inverted — lower time = better).
"""

import math
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


TEAM_NAME_MAP: Dict[str, str] = {
    "Alemania":              "Germany",
    "Arabia Saudí":          "Saudi Arabia",
    "Argelia":               "Algeria",
    "Argentina":             "Argentina",
    "Australia":             "Australia",
    "Austria":               "Austria",
    "Bosnia y Herzegovina":  "Bosnia and Herzegovina",
    "Brasil":                "Brazil",
    "Bélgica":               "Belgium",
    "Canadá":                "Canada",
    "Catar":                 "Qatar",
    "Chequia":               "Czechia",
    "Colombia":              "Colombia",
    "Costa de Marfil":       "Ivory Coast",
    "Croacia":               "Croatia",
    "Curazao":               "Curacao",
    "EE. UU.":               "United States",
    "Ecuador":               "Ecuador",
    "Egipto":                "Egypt",
    "Escocia":               "Scotland",
    "España":                "Spain",
    "Francia":               "France",
    "Ghana":                 "Ghana",
    "Haití":                 "Haiti",
    "Inglaterra":            "England",
    "Irak":                  "Iraq",
    "Islas de Cabo Verde":   "Cape Verde",
    "Japón":                 "Japan",
    "Jordania":              "Jordan",
    "Marruecos":             "Morocco",
    "México":                "Mexico",
    "Noruega":               "Norway",
    "Nueva Zelanda":         "New Zealand",
    "Panamá":                "Panama",
    "Paraguay":              "Paraguay",
    "Países Bajos":          "Netherlands",
    "Portugal":              "Portugal",
    "RD Congo":              "DR Congo",
    "RI de Irán":            "Iran",
    "República de Corea":    "South Korea",
    "Senegal":               "Senegal",
    "Sudáfrica":             "South Africa",
    "Suecia":                "Sweden",
    "Suiza":                 "Switzerland",
    "Turquía":               "Turkey",
    "Túnez":                 "Tunisia",
    "Uruguay":               "Uruguay",
    "Uzbekistán":            "Uzbekistan",
}

_RATIO_MIN = 0.50
_RATIO_MAX = 2.00


def _clamp(v: float) -> float:
    return max(_RATIO_MIN, min(_RATIO_MAX, v))


def _safe_ratio(
    val: float,
    avg: float,
    invert: bool = False,
    smooth: float = 0.0,
) -> float:
    """team_val / tournament_avg, clamped to [0.5, 2.0].
    smooth: additive Laplace smoothing for low-count stats.
    invert: use avg/val — for metrics where lower is better (e.g. recovery time).
    """
    denom = avg + smooth
    if denom == 0:
        return 1.0
    r = (val + smooth) / denom
    if invert:
        r = 1.0 / r if r > 0 else 1.0
    return _clamp(r)


def _geomean(values: list) -> float:
    vals = [v for v in values if v > 0]
    if not vals:
        return 1.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _parse_eff(s) -> float:
    """'1.36x' → 1.36. Returns 1.0 on parse failure."""
    try:
        return float(str(s).replace("x", "").strip())
    except (ValueError, AttributeError):
        return 1.0


def _find_data_dir(base_dir: str) -> str:
    """Return the most recently modified phase subfolder, or base_dir if none exist."""
    p = Path(base_dir)
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    if not subdirs:
        return str(p)
    return str(max(subdirs, key=lambda d: d.stat().st_mtime))


def _get(df_idx: pd.DataFrame, team: str, col: str) -> Optional[float]:
    """Safe scalar lookup; handles duplicate index rows gracefully."""
    if team not in df_idx.index:
        return None
    val = df_idx.at[team, col]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def load_latest_fifa_data(
    base_dir: str,
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Load and normalize FIFA stats from the latest available phase folder.

    Returns dict[english_team_name -> {attack_factor, defense_factor}]
    where 1.0 = tournament average for both factors.
    Returns None if any CSV fails to load.
    """
    try:
        data_dir = _find_data_dir(base_dir)

        atq_df = pd.read_csv(os.path.join(data_dir, "estadisticas_ataque.csv"))
        prt_df = pd.read_csv(os.path.join(data_dir, "porteria.csv"))
        dfn_df = pd.read_csv(os.path.join(data_dir, "defensa.csv"))
        mov_df = pd.read_csv(os.path.join(data_dir, "movimiento.csv"))

        atq_df["eff_parsed"] = atq_df["Efect_goles_prev"].apply(_parse_eff)
        for df in [atq_df, prt_df, dfn_df, mov_df]:
            df["team_en"] = df["Equipo"].map(TEAM_NAME_MAP)

        avg = {
            "xg":       atq_df["Goles_prev"].mean(),
            "xg_eff":   atq_df["eff_parsed"].mean(),
            "poss":     atq_df["Posesion_pct"].mean(),
            "saves":    prt_df["Paradas_portera"].mean(),
            "cs":       prt_df["Porterias_a_cero"].mean(),
            "press":    dfn_df["Presiones_defensivas"].mean(),
            "recovery": dfn_df["Tiempo_recuperacion_balon_s"].mean(),
            "runs":     mov_df["Desmarques_espalda_defensa"].mean(),
            "recv":     mov_df["Recepciones_bajo_presion"].mean(),
        }

        atq = atq_df.dropna(subset=["team_en"]).set_index("team_en")
        prt = prt_df.dropna(subset=["team_en"]).set_index("team_en")
        dfn = dfn_df.dropna(subset=["team_en"]).set_index("team_en")
        mov = mov_df.dropna(subset=["team_en"]).set_index("team_en")

        all_teams = (
            set(atq.index) | set(prt.index) | set(dfn.index) | set(mov.index)
        )

        result: Dict[str, Dict[str, float]] = {}

        for team in all_teams:
            # ── Attack pool ────────────────────────────────────────────────
            xg_v   = _get(atq, team, "Goles_prev")
            eff_v  = _get(atq, team, "eff_parsed")
            poss_v = _get(atq, team, "Posesion_pct")
            runs_v = _get(mov, team, "Desmarques_espalda_defensa")
            recv_v = _get(mov, team, "Recepciones_bajo_presion")

            attack_factor = _geomean([
                _safe_ratio(xg_v,   avg["xg"])     if xg_v   is not None else 1.0,
                _safe_ratio(eff_v,  avg["xg_eff"]) if eff_v  is not None else 1.0,
                _safe_ratio(poss_v, avg["poss"])   if poss_v is not None else 1.0,
                _safe_ratio(runs_v, avg["runs"])   if runs_v is not None else 1.0,
                _safe_ratio(recv_v, avg["recv"])   if recv_v is not None else 1.0,
            ])

            # ── Defense pool ───────────────────────────────────────────────
            saves_v    = _get(prt, team, "Paradas_portera")
            cs_v       = _get(prt, team, "Porterias_a_cero")
            press_v    = _get(dfn, team, "Presiones_defensivas")
            recovery_v = _get(dfn, team, "Tiempo_recuperacion_balon_s")

            defense_factor = _geomean([
                _safe_ratio(saves_v,    avg["saves"],    smooth=1.0)          if saves_v    is not None else 1.0,
                _safe_ratio(cs_v,       avg["cs"],       smooth=1.0)          if cs_v       is not None else 1.0,
                _safe_ratio(press_v,    avg["press"])                         if press_v    is not None else 1.0,
                _safe_ratio(recovery_v, avg["recovery"], invert=True)         if recovery_v is not None else 1.0,
            ])

            result[team] = {
                "attack_factor":  round(attack_factor, 4),
                "defense_factor": round(defense_factor, 4),
            }

        return result

    except Exception:
        return None
