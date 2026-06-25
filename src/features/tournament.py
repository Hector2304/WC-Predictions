"""
Tournament features.

is_world_cup  : 1 for FIFA World Cup proper (group stage + knockout).
                Excludes Viva World Cup, CONIFA, qualifiers, friendlies.
is_knockout   : 1 for FIFA World Cup knockout phase only (R16 onward).
                Subset of is_world_cup; phase dates from wc_phases.py.
is_major_tourn: backward-compat alias for is_world_cup (kept so v3 predict
                path and any script importing this column continues to work).

Other tournament dummies (is_euros, is_copa_am, etc.) are computed for
completeness and used by v4, but are not part of the v5 feature set.
"""

import pandas as pd

from src.features.wc_phases import is_wc_knockout

# v5 uses only these two
WC_FEATURES = ["is_world_cup", "is_knockout"]

# v4 dummies — kept for backward compat
TOURNAMENT_DUMMIES = [
    "is_world_cup",
    "is_euros",
    "is_copa_am",
    "is_afcon",
    "is_asian_cup",
    "is_gold_cup",
]

_EXCLUDE = ["qual", "friendly"]

_RULES: list[tuple[str, list[str]]] = [
    ("is_world_cup",  ["fifa world cup"]),
    ("is_euros",      ["uefa euro"]),
    ("is_copa_am",    ["copa am", "copa ám"]),
    ("is_afcon",      ["african cup of nations"]),
    ("is_asian_cup",  ["afc asian cup"]),
    ("is_gold_cup",   ["gold cup"]),
]


def _match(tournament: str, phrases: list[str]) -> int:
    t = tournament.lower()
    if any(ex in t for ex in _EXCLUDE):
        return 0
    return int(any(ph in t for ph in phrases))


def add_tournament_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, phrases in _RULES:
        out[col] = df["tournament"].apply(lambda t, p=phrases: _match(t, p)).astype(float)
    out["is_knockout"] = df.apply(
        lambda r: is_wc_knockout(r.tournament, r.date), axis=1
    ).astype(float)
    # backward compat: is_major_tourn = is_world_cup (clean, no false positives)
    out["is_major_tourn"] = out["is_world_cup"]
    return out
