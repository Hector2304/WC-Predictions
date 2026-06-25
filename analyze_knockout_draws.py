"""
Empirical knockout draw rates by WC edition (1986-2022).

Uses wc_phases.py phase dates to isolate knockout matches per edition
and computes draw rate at 90 minutes (home_score == away_score).

Looks for structural breaks around:
  1998 — format change to 32 teams
  2002 — last edition with golden-goal rule (sudden-death extra time)

Output informs the choice of training window for theta_D_knockout
calibration in the next step.

Run:
    py analyze_knockout_draws.py
"""

import pandas as pd
from src.features.wc_phases import WC_KNOCKOUT_START

df = pd.read_csv("international_results-master/results.csv")
df["date"] = pd.to_datetime(df["date"])
wc = df[df["tournament"] == "FIFA World Cup"].copy()
wc["year"] = wc["date"].dt.year

NOTES = {
    1986: "24-team, R16 introduced",
    1990: "24-team",
    1994: "24-team, last 24-team edition",
    1998: "32-team, golden goal",
    2002: "32-team, golden goal (last edition)",
    2006: "32-team",
    2010: "32-team",
    2014: "32-team",
    2018: "32-team",
    2022: "32-team (Qatar)",
}

rows = []
print()
print(f"  {'year':<6}  {'n_ko':>5}  {'draws':>6}  {'draw_rate':>10}  note")
print("  " + "-" * 65)

for yr in sorted(WC_KNOCKOUT_START):
    ko_start = WC_KNOCKOUT_START[yr]
    edition  = wc[wc["year"] == yr]
    knockout = edition[edition["date"] >= ko_start]
    n     = len(knockout)
    draws = int((knockout["home_score"] == knockout["away_score"]).sum())
    rate  = draws / n if n else 0.0
    rows.append({"year": yr, "n": n, "draws": draws, "rate": rate})
    note = NOTES.get(yr, "")
    print(f"  {yr:<6}  {n:>5}  {draws:>6}  {rate:>10.1%}  {note}")

print()

df_rows = pd.DataFrame(rows)

eras = [
    ("24-team (1986-1994)",        [1986, 1990, 1994]),
    ("Golden goal (1998-2002)",    [1998, 2002]),
    ("Modern 32-team (2006-2022)", [2006, 2010, 2014, 2018, 2022]),
    ("Post-1998 combined",         [1998, 2002, 2006, 2010, 2014, 2018, 2022]),
    ("Full window (1986-2022)",    list(WC_KNOCKOUT_START.keys())),
]

print("  ERA SUMMARY")
print("  " + "-" * 55)
for label, years in eras:
    sub = df_rows[df_rows["year"].isin(years)]
    if sub.empty:
        continue
    n = sub["n"].sum()
    d = sub["draws"].sum()
    print(f"  {label:<32}  {n:>3} matches  {d}/{n} draws  {d/n:.1%}")

print()
print("  ROUND BREAKDOWN (pooled 1998-2022)")
print("  " + "-" * 55)

# Round-level breakdown: what's the draw rate by round?
# We can approximate by match number within each edition's knockout.
# For 32-team editions: R16=matches 1-8, QF=9-12, SF=13-14, Final=16
# (match 15 is 3rd place playoff)
for yr in [y for y in sorted(WC_KNOCKOUT_START) if y >= 1998]:
    ko_start = WC_KNOCKOUT_START[yr]
    edition  = wc[wc["year"] == yr]
    knockout = edition[edition["date"] >= ko_start].sort_values("date")
    # tag round by position (0-indexed)
    knockout = knockout.reset_index(drop=True)
    knockout["draw"] = knockout["home_score"] == knockout["away_score"]

# Pool all editions 1998-2022 and assign round by sorted position
all_rounds = {"R16": [], "QF": [], "SF": [], "3rd/Final": []}
for yr in [y for y in sorted(WC_KNOCKOUT_START) if y >= 1998]:
    ko_start = WC_KNOCKOUT_START[yr]
    edition  = wc[wc["year"] == yr]
    knockout = edition[edition["date"] >= ko_start].sort_values("date").reset_index(drop=True)
    draws = (knockout["home_score"] == knockout["away_score"]).values
    n = len(knockout)
    if n == 16:
        for i, d in enumerate(draws):
            if i < 8:   all_rounds["R16"].append(d)
            elif i < 12: all_rounds["QF"].append(d)
            elif i < 14: all_rounds["SF"].append(d)
            else:        all_rounds["3rd/Final"].append(d)

for rnd, vals in all_rounds.items():
    if not vals:
        continue
    n = len(vals); d = sum(vals)
    print(f"  {rnd:<15}  {n:>3} matches  {d}/{n} draws  {d/n:.1%}")
