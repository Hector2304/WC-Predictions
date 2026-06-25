"""
Side-by-side comparison of v4 vs v5 on WC-specific and overall metrics.

v4: 6 tournament dummies (is_world_cup + 5 others)
v5: is_world_cup + is_knockout

Primary question: does is_knockout close the draw calibration gap on
WC matches better than the broader tournament dummies?

Evaluation windows:
  - Overall test set (2023+)           : 3600+ matches, all tournaments
  - WC group stage test (2026)         : ~44 matches, no knockout yet
  - WC knockout retrospective (2022)   : from val set — note: used for
                                         theta_D calibration, not held out

Run from the project root:
    py compare_v4_v5.py
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.data.load import load_results
from src.features.elo import compute_elo
from src.features.h2h import compute_h2h
from src.features.form import compute_form
from src.features.tournament import add_tournament_features

TEST_START = "2023-01-01"
VAL_START  = "2022-01-01"
CLASSES    = ["H", "D", "A"]


# ── helpers ───────────────────────────────────────────────────────────────────

def label_result(hs: int, as_: int) -> str:
    if hs > as_: return "H"
    if hs < as_: return "A"
    return "D"


def apply_threshold(proba: np.ndarray, theta_d: float) -> np.ndarray:
    p_h, p_d, p_a = proba[:, 0], proba[:, 1], proba[:, 2]
    return np.where(p_d > theta_d, "D", np.where(p_h >= p_a, "H", "A"))


def log_loss(y_true, y_proba):
    idx = np.array([CLASSES.index(y) for y in y_true])
    p   = y_proba[np.arange(len(y_true)), idx]
    return float(-np.mean(np.log(np.maximum(p, 1e-15))))


def metrics_row(label, y, pred, proba):
    if len(y) == 0:
        return {"model": label, "n": 0}
    return {
        "model":     label,
        "n":         len(y),
        "log_loss":  round(log_loss(y, proba), 4),
        "accuracy":  round(accuracy_score(y, pred), 3),
        "f1_draw":   round(f1_score(y, pred, labels=["D"], average="macro", zero_division=0), 3),
        "draw_pred": round((pred == "D").mean(), 3),
        "draw_act":  round((y == "D").mean(), 3),
    }


# ── load ──────────────────────────────────────────────────────────────────────
with open("models/v4_config.json") as f: cfg4 = json.load(f)
with open("models/v5_config.json") as f: cfg5 = json.load(f)
m4 = joblib.load("models/poisson_dc_v4.joblib")
m5 = joblib.load("models/poisson_dc_v5.joblib")

df_raw = load_results()
p      = cfg5["priors"]
hp     = cfg5["hyperparams"]

df, _  = compute_elo(df_raw, home_advantage=cfg5["home_advantage"])
df["result"] = df.apply(lambda r: label_result(r.home_score, r.away_score), axis=1)
df = compute_h2h(df, p["global_home_avg"], p["global_away_avg"], k=hp["h2h_k"])
df = compute_form(df, p["global_avg"], window=hp["form_window"], k=hp["form_k"])
df = add_tournament_features(df)

val  = df[(df["date"] >= VAL_START) & (df["date"] < TEST_START)].copy()
test = df[df["date"] >= TEST_START].copy()

def eval_split(split, cfg4, cfg5):
    p4 = m4.predict_proba(split)
    p5 = m5.predict_proba(split)
    d4 = apply_threshold(p4, cfg4["theta_D"])
    d5 = apply_threshold(p5, cfg5["theta_D"])
    y  = split["result"].values
    return p4, p5, d4, d5, y


print(f"v4: theta_D={cfg4['theta_D']}  rho={cfg4['rho']}")
print(f"v5: theta_D={cfg5['theta_D']}  rho={cfg5['rho']}")
print()

# ── 1. Overall test set ───────────────────────────────────────────────────────
p4t, p5t, d4t, d5t, yt = eval_split(test, cfg4, cfg5)
print("-- 1. Overall (test >= 2023, all tournaments) ---------------------------")
overall = pd.DataFrame([
    metrics_row("v4", yt, d4t, p4t),
    metrics_row("v5", yt, d5t, p5t),
])
print(overall.to_string(index=False))
print()

# ── 2. WC group stage (2026, test set) ───────────────────────────────────────
wc_test = test[(test["is_world_cup"] == 1) & (test["is_knockout"] == 0)].copy()
p4wg, p5wg, d4wg, d5wg, ywg = eval_split(wc_test, cfg4, cfg5)
print("-- 2. WC 2026 group stage (test set — no knockout yet) ------------------")
wc_group = pd.DataFrame([
    metrics_row("v4", ywg, d4wg, p4wg),
    metrics_row("v5", ywg, d5wg, p5wg),
])
print(wc_group.to_string(index=False))
print()

# ── 3. WC knockout retrospective (2022 val set — used for theta_D calib) ─────
wc_ko_val = val[(val["is_world_cup"] == 1) & (val["is_knockout"] == 1)].copy()
wc_gp_val = val[(val["is_world_cup"] == 1) & (val["is_knockout"] == 0)].copy()
print("-- 3. WC 2022 retrospective (val set — not fully held out) --------------")
print("   NOTE: val set was used to calibrate theta_D — treat as indicative only")
print()
print("   WC 2022 group stage:")
p4kg, p5kg, d4kg, d5kg, ykg = eval_split(wc_gp_val, cfg4, cfg5)
ko_group = pd.DataFrame([
    metrics_row("v4", ykg, d4kg, p4kg),
    metrics_row("v5", ykg, d5kg, p5kg),
])
print(ko_group.to_string(index=False))
print()
print("   WC 2022 knockout (R16 through Final):")
p4ko, p5ko, d4ko, d5ko, yko = eval_split(wc_ko_val, cfg4, cfg5)
ko_rows = pd.DataFrame([
    metrics_row("v4", yko, d4ko, p4ko),
    metrics_row("v5", yko, d5ko, p5ko),
])
print(ko_rows.to_string(index=False))
print()

# ── 4. v5 coefficients ────────────────────────────────────────────────────────
print("-- 4. v5 feature coefficients -------------------------------------------")
coef = m5.coef_summary()
wc_cols = [c for c in coef.columns if c in ["is_world_cup", "is_knockout"]]
base_cols = ["target", "intercept", "elo_diff_scaled", "neutral",
             "h2h_home_goals_mu", "h2h_away_goals_mu"] + wc_cols
print(coef[[c for c in base_cols if c in coef.columns]].to_string(
    index=False, float_format="{:.4f}".format
))
print()
print("  Interpretation:")
print(f"  WC group stage extra effect: home {coef.loc[coef['target']=='home','is_world_cup'].values[0]:+.4f}  "
      f"away {coef.loc[coef['target']=='away','is_world_cup'].values[0]:+.4f}")
ko_h = coef.loc[coef['target']=='home','is_knockout'].values[0]
ko_a = coef.loc[coef['target']=='away','is_knockout'].values[0]
print(f"  WC knockout extra effect:    home {ko_h:+.4f}  away {ko_a:+.4f}")
print(f"  Combined knockout effect:    home {coef.loc[coef['target']=='home','is_world_cup'].values[0]+ko_h:+.4f}  "
      f"away {coef.loc[coef['target']=='away','is_world_cup'].values[0]+ko_a:+.4f}")
print()

# ── 5. Draw rate gap: where does each model predict draws ─────────────────────
print("-- 5. Draw rate gap by tournament category (test set) -------------------")
def tc(t):
    t = t.lower()
    if "fifa world cup" in t and "qual" not in t: return "World Cup"
    if "qual" in t:           return "Qualifier"
    if "nations league" in t: return "Nations League"
    if "friendly" in t:       return "Friendly"
    return "Other"

test["cat"] = test["tournament"].apply(tc)
print(f"  {'category':<20} {'n':>5}  {'draw_act':>9}  {'draw_v4':>9}  {'draw_v5':>9}")
print("  " + "-" * 62)
for cat, grp in test.groupby("cat"):
    idx = grp.index
    mask = np.isin(np.arange(len(test)), [test.index.get_loc(i) for i in idx])
    y_c  = yt[test["cat"].values == cat]
    d4_c = d4t[test["cat"].values == cat]
    d5_c = d5t[test["cat"].values == cat]
    dr_r = (y_c == "D").mean()
    dr4  = (d4_c == "D").mean()
    dr5  = (d5_c == "D").mean()
    print(f"  {cat:<20} {len(y_c):>5}  {dr_r:>9.1%}  {dr4:>9.1%}  {dr5:>9.1%}")
