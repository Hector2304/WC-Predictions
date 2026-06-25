import pandas as pd
from pathlib import Path

_RAW = Path(__file__).parents[2] / "international_results-master"


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
    df = df.sort_values("date").reset_index(drop=True)
    return df
