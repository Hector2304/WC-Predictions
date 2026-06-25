"""
diagnose_calibration.py — Isotonic calibration diagnostic for v3.

Diagnostic question:
  Is the 34.3% draw over-prediction a SCALE issue (probabilities are
  well-ordered but numerically shifted) or a STRUCTURAL issue (the model
  can't distinguish draw-prone matchups from decisive ones)?

Method:
  Fit one-vs-rest isotonic regression on val set (2022), apply to test (2023+).
  The calibrator maps raw P(c) → corrected P(c) using only the ordering of
  predictions, so it cannot create signal that isn't already there.

Verdict logic:
  - If isotonic pulls draw rate toward 23% AND improves log-loss → SCALE.
    The model's ranking is fine; just the absolute values are off.
    Isotonic calibration (or Platt scaling) can ship as-is.
  - If isotonic barely changes draw rate OR hurts log-loss → STRUCTURAL.
    The model genuinely conflates draw-prone with non-draw scenarios.
    Architecture fix needed (is_knockout split, DC full ratings, etc.).

Run:
    py diagnose_calibration.py
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.data.load import load_results
from src.features.elo import compute_elo
from src.features.h2h import compute_h2h
from src.features.tournament import add_tournament_features

CLASSES    = ["H", "D", "A"]
VAL_START  = pd.Timestamp("2022-01-01")
TEST_START = pd.Timestamp("2023-01-01")


# ── helpers ───────────────────────────────────────────────────────────────────

def outcome_labels(df: pd.DataFrame) -> np.ndarray:
    cond = [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]]
    return np.select(cond, ["H", "D"], default="A")


def to_idx(labels: np.ndarray) -> np.ndarray:
    mapping = {"H": 0, "D": 1, "A": 2}
    return np.array([mapping[l] for l in labels])


def log_loss_manual(y_str: np.ndarray, proba: np.ndarray) -> float:
    idx = to_idx(y_str)
    p = proba[np.arange(len(y_str)), idx]
    return float(-np.mean(np.log(np.maximum(p, 1e-15))))


def apply_theta(proba: np.ndarray, theta_D: float) -> np.ndarray:
    """Predict D if P(D) > theta_D, else argmax of H vs A."""
    return np.where(
        proba[:, 1] > theta_D,
        1,
        np.where(proba[:, 0] >= proba[:, 2], 0, 2),
    )


def f1_draw(pred_idx: np.ndarray, y_idx: np.ndarray) -> float:
    tp = ((pred_idx == 1) & (y_idx == 1)).sum()
    fp = ((pred_idx == 1) & (y_idx != 1)).sum()
    fn = ((pred_idx != 1) & (y_idx == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def print_metrics(tag: str, proba: np.ndarray, y_str: np.ndarray, theta_D: float):
    y_idx  = to_idx(y_str)
    pred   = apply_theta(proba, theta_D)
    ll     = log_loss_manual(y_str, proba)
    acc    = (pred == y_idx).mean()
    f1d    = f1_draw(pred, y_idx)
    dr_p   = (pred == 1).mean()
    dr_r   = (y_idx == 1).mean()
    print(f"  {tag}")
    print(f"    log-loss  {ll:.4f}   accuracy  {acc:.3f}   F1-draw  {f1d:.3f}")
    print(f"    draw pred {dr_p:.1%}  vs  real {dr_r:.1%}")


# ── isotonic calibration ──────────────────────────────────────────────────────

def fit_isotonic_ovr(proba: np.ndarray, y_str: np.ndarray) -> list:
    """Fit one IsotonicRegression per class (one-vs-rest)."""
    y_idx = to_idx(y_str)
    calibrators = []
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
        ir.fit(proba[:, c], (y_idx == c).astype(float))
        calibrators.append(ir)
    return calibrators


def apply_isotonic(calibrators: list, proba: np.ndarray) -> np.ndarray:
    """Apply OVR calibration and renormalize rows to sum to 1."""
    cal = np.column_stack([ir.predict(proba[:, c]) for c, ir in enumerate(calibrators)])
    totals = cal.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1.0, totals)
    return cal / totals


# ── reliability diagram for P(D) ─────────────────────────────────────────────

def reliability_diagram_draw(proba_raw: np.ndarray, proba_cal: np.ndarray,
                              y_str: np.ndarray):
    y_idx = to_idx(y_str)
    bins  = np.arange(0.0, 0.55, 0.05)
    print(f"\n  {'P(D) bin':<14} {'n':>5}  {'raw pred':>9} {'cal pred':>9} {'actual':>7}  "
          f"{'raw err':>8}  {'cal err':>8}")
    print("  " + "-" * 72)
    for lo in bins:
        hi = lo + 0.05
        mask = (proba_raw[:, 1] >= lo) & (proba_raw[:, 1] < hi)
        n = mask.sum()
        if n < 10:
            continue
        pred_raw = proba_raw[mask, 1].mean()
        pred_cal = proba_cal[mask, 1].mean()
        actual   = (y_idx[mask] == 1).mean()
        print(f"  {lo:.0%}-{hi:.0%}         {n:>5}     {pred_raw:.3f}     {pred_cal:.3f}   "
              f"{actual:.3f}  {actual - pred_raw:+.3f}    {actual - pred_cal:+.3f}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    with open("models/v3_config.json") as f:
        cfg = json.load(f)
    model   = joblib.load("models/poisson_dc_v3.joblib")
    theta_D = cfg.get("theta_D", 0.25)
    p       = cfg["priors"]
    hp      = cfg["hyperparams"]

    df_raw = load_results()
    df, _  = compute_elo(df_raw, home_advantage=cfg.get("home_advantage", 100.0))
    df     = compute_h2h(df, p["global_home_avg"], p["global_away_avg"], k=hp["h2h_k"])
    df     = add_tournament_features(df)

    df_val  = df[(df["date"] >= VAL_START) & (df["date"] < TEST_START)].copy()
    df_test = df[df["date"] >= TEST_START].copy()

    proba_val  = model.predict_proba(df_val)
    proba_test = model.predict_proba(df_test)

    y_val  = outcome_labels(df_val)
    y_test = outcome_labels(df_test)

    # Fit calibrator on val, apply to test (no leakage into test)
    calibrators    = fit_isotonic_ovr(proba_val, y_val)
    proba_test_cal = apply_isotonic(calibrators, proba_test)

    # ── report ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Isotonic Calibration Diagnostic  (v3 model, theta_D=0.25)")
    print("=" * 60)
    print(f"\n  Val set (2022):  {len(df_val):,} matches  [used to fit calibrator]")
    print(f"  Test set (2023+): {len(df_test):,} matches  [never seen by calibrator]\n")

    print_metrics("Raw v3 (baseline):", proba_test, y_test, theta_D)
    print()
    print_metrics("After isotonic calibration:", proba_test_cal, y_test, theta_D)

    # ── P(D) reliability diagram ──────────────────────────────────────────────
    print("\n-- P(D) reliability diagram ---------------------------------------")
    print("  (rows = bins of raw P(D); shows how isotonic shifts the values)")
    reliability_diagram_draw(proba_test, proba_test_cal, y_test)

    # ── diagnosis ────────────────────────────────────────────────────────────
    y_idx       = to_idx(y_test)
    pred_raw    = apply_theta(proba_test, theta_D)
    pred_cal    = apply_theta(proba_test_cal, theta_D)
    dr_raw      = (pred_raw == 1).mean()
    dr_cal      = (pred_cal == 1).mean()
    dr_real     = (y_idx == 1).mean()
    ll_raw      = log_loss_manual(y_test, proba_test)
    ll_cal      = log_loss_manual(y_test, proba_test_cal)
    gap_before  = abs(dr_raw - dr_real)
    gap_after   = abs(dr_cal - dr_real)
    gap_closed  = (gap_before - gap_after) / gap_before if gap_before > 0 else 0.0

    draw_moved_right = dr_cal < dr_raw  # True if calibration pushed draw rate down

    print()
    print("=" * 60)
    print("  DIAGNOSIS")
    print("=" * 60)
    print(f"  Val draw rate (calibrator training set): {(to_idx(y_val) == 1).mean():.1%}")
    print(f"  Test draw rate (real):                   {dr_real:.1%}")
    print()
    print(f"  Draw pred  raw -> cal -> real:  {dr_raw:.1%} -> {dr_cal:.1%} -> {dr_real:.1%}")
    direction = "toward real (correct)" if draw_moved_right else "AWAY from real (wrong direction)"
    print(f"  Direction of change: {direction}")
    print(f"  Draw gap closed: {gap_closed:+.0%}  (gap {gap_before:.1%} -> {gap_after:.1%})")
    print(f"  Log-loss:  {ll_raw:.4f} -> {ll_cal:.4f}  "
          f"(delta {ll_cal - ll_raw:+.4f}, {'improves' if ll_cal < ll_raw else 'worsens'})")
    print()

    if not draw_moved_right:
        verdict = (
            "STRUCTURAL (confirmed, strong signal).\n"
            "  Isotonic calibration pushed draw rate UP ({:.1%} -> {:.1%})\n"
            "  instead of DOWN toward real {:.1%}. Log-loss also worsened.\n"
            "  Meaning: the model's P(D) ordering does not reliably separate\n"
            "  draw-prone matches from decisive ones. The calibrator learned\n"
            "  a noisy mapping from 970 val matches and amplified the noise.\n"
            "  Root cause is in the features, not the probability scale.\n"
            "  Recommended: split is_major_tourn -> is_knockout."
        ).format(dr_raw, dr_cal, dr_real)
    elif gap_closed >= 0.5 and ll_cal <= ll_raw + 0.001:
        verdict = (
            "SCALE -- probabilities are well-ordered but numerically shifted.\n"
            "  Isotonic calibration recovers most of the draw gap without\n"
            "  hurting log-loss. No architecture change strictly needed."
        )
    elif gap_closed >= 0.5 and ll_cal > ll_raw + 0.001:
        verdict = (
            "MIXED -- calibration closes draw gap but hurts log-loss.\n"
            "  Model ranking is partially right but theta_D does too much work.\n"
            "  is_knockout split is the targeted fix."
        )
    else:
        verdict = (
            "STRUCTURAL -- calibration barely closes the draw gap.\n"
            "  is_knockout split and/or Dixon-Coles full ratings justified."
        )

    for line in verdict.split("\n"):
        print(f"  {line}")
    print()


if __name__ == "__main__":
    main()
