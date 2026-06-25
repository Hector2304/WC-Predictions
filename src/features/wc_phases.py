"""
World Cup phase lookup: group stage vs knockout.

Knockout start dates derived directly from the martj42 dataset by counting
group stage matches per edition (36 for 24-team, 48 for 32-team, 72 for
48-team) and reading the date of the first match past that count.

Coverage: 1986-2022 (editions with a distinct R16 knockout format).
Pre-1986 WC matches are tagged is_knockout=0 by default.
2026 knockout: not yet in the dataset; predict.py passes is_knockout directly.
"""

import pandas as pd

# {edition_year: first date of knockout phase}
# Derived from dataset — do not edit without re-running the derivation query.
WC_KNOCKOUT_START: dict[int, pd.Timestamp] = {
    1986: pd.Timestamp("1986-06-15"),
    1990: pd.Timestamp("1990-06-23"),
    1994: pd.Timestamp("1994-07-02"),
    1998: pd.Timestamp("1998-06-27"),
    2002: pd.Timestamp("2002-06-15"),
    2006: pd.Timestamp("2006-06-24"),
    2010: pd.Timestamp("2010-06-26"),
    2014: pd.Timestamp("2014-06-28"),
    2018: pd.Timestamp("2018-06-30"),
    2022: pd.Timestamp("2022-12-03"),
}


def is_wc_knockout(tournament: str, date: pd.Timestamp) -> int:
    """Return 1 if the match is a FIFA World Cup knockout game, else 0."""
    if "fifa world cup" not in tournament.lower():
        return 0
    ko_start = WC_KNOCKOUT_START.get(date.year)
    if ko_start is None:
        return 0
    return int(date >= ko_start)
