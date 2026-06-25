import pandas as pd
from pathlib import Path

_RAW = Path(__file__).parents[2] / "international_results-master"

# Historical name → current FIFA name used in mundial2026.csv
_TEAM_ALIASES: dict[str, str] = {
    "Czech Republic":  "Czechia",
    "Cura\xc3\xa7ao":  "Curacao",  # mojibake for Curaçao in results.csv (double-encoded UTF-8)
    "Cura\xe7ao":      "Curacao",  # correctly-encoded Curaçao (fallback)
}


def load_results() -> pd.DataFrame:
    df = pd.read_csv(_RAW / "results.csv", parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    if df["neutral"].dtype == object:
        df["neutral"] = (
            df["neutral"].str.strip().str.upper()
            .map({"TRUE": True, "FALSE": False})
            .fillna(False)
            .astype(bool)
        )
    else:
        df["neutral"] = df["neutral"].astype(bool)
    df["home_team"] = df["home_team"].replace(_TEAM_ALIASES)
    df["away_team"] = df["away_team"].replace(_TEAM_ALIASES)
    df = df.sort_values("date").reset_index(drop=True)
    return df
