import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson as poisson_dist
from sklearn.linear_model import PoissonRegressor
from typing import List, Optional, Tuple

MAX_GOALS = 7

# Default feature sets — always elo_diff (scaled) + neutral.
# Pass extra_home / extra_away to add columns from the DataFrame.
_BASE_FEATURES = ["elo_diff_scaled", "neutral"]


class PoissonDC:
    """
    Two Poisson regressions for home/away goals with Dixon-Coles correction.

    Base features: elo_diff_scaled (= elo_diff/400), neutral (0/1).
    Additional features can be supplied per model via extra_home / extra_away.
    """

    def __init__(
        self,
        extra_home: Optional[List[str]] = None,
        extra_away: Optional[List[str]] = None,
        alpha: float = 0.0,
    ):
        self.extra_home = list(extra_home or [])
        self.extra_away = list(extra_away or [])
        self._home_model = PoissonRegressor(alpha=alpha, fit_intercept=True, max_iter=300)
        self._away_model = PoissonRegressor(alpha=alpha, fit_intercept=True, max_iter=300)
        self.rho_: float = 0.0

    # ── public API ────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "PoissonDC":
        df = _ensure_scaled(df)
        self._home_model.fit(self._home_X(df), df["home_score"].values)
        self._away_model.fit(self._away_X(df), df["away_score"].values)

        lam_h = self._home_model.predict(self._home_X(df))
        lam_a = self._away_model.predict(self._away_X(df))
        self.rho_ = _fit_rho(
            df["home_score"].values.astype(int),
            df["away_score"].values.astype(int),
            lam_h, lam_a,
        )
        return self

    def predict_lambdas(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = _ensure_scaled(df)
        return self._home_model.predict(self._home_X(df)), \
               self._away_model.predict(self._away_X(df))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns (n, 3) array: columns [P(H), P(D), P(A)]."""
        lam_h, lam_a = self.predict_lambdas(df)
        return _proba_from_joint(_joint_matrix(lam_h, lam_a, self.rho_))

    def predict_scoreline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns most probable (pred_home, pred_away) per match."""
        lam_h, lam_a = self.predict_lambdas(df)
        joint = _joint_matrix(lam_h, lam_a, self.rho_)
        G = MAX_GOALS + 1
        flat_idx = joint.reshape(len(lam_h), -1).argmax(axis=1)
        return pd.DataFrame(
            {"pred_home": flat_idx // G, "pred_away": flat_idx % G},
            index=df.index,
        )

    def predict_from_lambdas(
        self, lam_h: np.ndarray, lam_a: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute (proba, score_home, score_away) from externally supplied lambdas.
        Applies the fitted Dixon-Coles rho correction.
        """
        joint = _joint_matrix(lam_h, lam_a, self.rho_)
        proba = _proba_from_joint(joint)
        G = MAX_GOALS + 1
        flat_idx = joint.reshape(len(lam_h), -1).argmax(axis=1)
        return proba, flat_idx // G, flat_idx % G

    def coef_summary(self) -> pd.DataFrame:
        home_names = _BASE_FEATURES + self.extra_home
        away_names = _BASE_FEATURES + self.extra_away
        rows = []
        for label, model, names in [
            ("home", self._home_model, home_names),
            ("away", self._away_model, away_names),
        ]:
            row: dict = {"target": label, "intercept": model.intercept_}
            for name, coef in zip(names, model.coef_):
                row[name] = coef
            rows.append(row)
        return pd.DataFrame(rows)

    # ── internals ─────────────────────────────────────────────────────

    def _home_X(self, df: pd.DataFrame) -> np.ndarray:
        cols = _BASE_FEATURES + self.extra_home
        return df[cols].values.astype(float)

    def _away_X(self, df: pd.DataFrame) -> np.ndarray:
        cols = _BASE_FEATURES + self.extra_away
        return df[cols].values.astype(float)


# ── module-level helpers ──────────────────────────────────────────────

def _ensure_scaled(df: pd.DataFrame) -> pd.DataFrame:
    """Add elo_diff_scaled column if not already present."""
    if "elo_diff_scaled" not in df.columns:
        df = df.copy()
        df["elo_diff_scaled"] = df["elo_diff"] / 400.0
    return df


def _joint_matrix(lam_h: np.ndarray, lam_a: np.ndarray, rho: float) -> np.ndarray:
    """(n, G, G) joint probability matrix with Dixon-Coles correction."""
    G = MAX_GOALS + 1
    goals = np.arange(G)
    p_h = poisson_dist.pmf(goals[None, :], lam_h[:, None])
    p_a = poisson_dist.pmf(goals[None, :], lam_a[:, None])
    joint = p_h[:, :, None] * p_a[:, None, :]
    joint[:, 0, 0] *= np.maximum(1 - lam_h * lam_a * rho, 0)
    joint[:, 0, 1] *= np.maximum(1 + lam_h * rho, 0)
    joint[:, 1, 0] *= np.maximum(1 + lam_a * rho, 0)
    joint[:, 1, 1] *= np.maximum(1 - rho, 0)
    return joint


def _proba_from_joint(joint: np.ndarray) -> np.ndarray:
    G = MAX_GOALS + 1
    i_idx, j_idx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    p_home = (joint * (i_idx > j_idx)[None]).sum(axis=(1, 2))
    p_draw = (joint * (i_idx == j_idx)[None]).sum(axis=(1, 2))
    p_away = (joint * (i_idx < j_idx)[None]).sum(axis=(1, 2))
    total = (p_home + p_draw + p_away)[:, None]
    return np.column_stack([p_home, p_draw, p_away]) / total


def _fit_rho(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lam_h: np.ndarray,
    lam_a: np.ndarray,
) -> float:
    mask = (home_goals <= 1) & (away_goals <= 1)
    h, a, lh, la = home_goals[mask], away_goals[mask], lam_h[mask], lam_a[mask]

    def neg_ll(rho: float) -> float:
        tau = np.where(
            (h == 0) & (a == 0), np.maximum(1 - lh * la * rho, 1e-10),
            np.where(
                (h == 0) & (a == 1), np.maximum(1 + lh * rho, 1e-10),
                np.where(
                    (h == 1) & (a == 0), np.maximum(1 + la * rho, 1e-10),
                    np.maximum(1 - rho, 1e-10),
                ),
            ),
        )
        p_base = poisson_dist.pmf(h, lh) * poisson_dist.pmf(a, la)
        return -np.sum(np.log(np.maximum(p_base * tau, 1e-10)))

    return float(minimize_scalar(neg_ll, bounds=(-0.4, 0.4), method="bounded").x)
