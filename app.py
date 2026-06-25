"""
World Cup 2026 Predictor — Streamlit UI

Run from the project root:
    py -m streamlit run app.py
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.data.load import load_results
from src.features.elo import compute_elo
from src.features.h2h import compute_h2h
from src.features.tournament import add_tournament_features
from src.features.tournament_form import (
    form_adjusted_lambdas,
    form_adjusted_lambdas_v2,
    form_adjusted_lambdas_v3,
    get_wc_form,
    get_wc_form_v2,
)
from src.features.fifa_stats import load_latest_fifa_data

FIFA_DATA_DIR = "DATOS FIFA"

FLAGS: dict[str, str] = {
    "Algeria": "🇩🇿", "Argentina": "🇦🇷", "Australia": "🇦🇺",
    "Austria": "🇦🇹", "Belgium": "🇧🇪", "Bosnia and Herzegovina": "🇧🇦",
    "Brazil": "🇧🇷", "Canada": "🇨🇦", "Cape Verde": "🇨🇻",
    "Colombia": "🇨🇴", "Croatia": "🇭🇷", "Curacao": "🇨🇼",
    "Czechia": "🇨🇿", "DR Congo": "🇨🇩", "Ecuador": "🇪🇨",
    "Egypt": "🇪🇬", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "France": "🇫🇷",
    "Germany": "🇩🇪", "Ghana": "🇬🇭", "Haiti": "🇭🇹",
    "Iran": "🇮🇷", "Iraq": "🇮🇶", "Ivory Coast": "🇨🇮",
    "Japan": "🇯🇵", "Jordan": "🇯🇴", "Mexico": "🇲🇽",
    "Morocco": "🇲🇦", "Netherlands": "🇳🇱", "New Zealand": "🇳🇿",
    "Norway": "🇳🇴", "Panama": "🇵🇦", "Paraguay": "🇵🇾",
    "Portugal": "🇵🇹", "Qatar": "🇶🇦", "Saudi Arabia": "🇸🇦",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Senegal": "🇸🇳", "South Africa": "🇿🇦",
    "South Korea": "🇰🇷", "Spain": "🇪🇸", "Sweden": "🇸🇪",
    "Switzerland": "🇨🇭", "Tunisia": "🇹🇳", "Turkey": "🇹🇷",
    "United States": "🇺🇸", "Uruguay": "🇺🇾", "Uzbekistan": "🇺🇿",
}


def flag(team: str) -> str:
    return FLAGS.get(team, "🏳️")


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="centered",
)

# ── load pipeline (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_pipeline():
    with open("models/v5_config.json") as f:
        cfg = json.load(f)
    model = joblib.load("models/poisson_dc_v5.joblib")

    ha = cfg.get("home_advantage", 20.0)
    p  = cfg["priors"]
    hp = cfg["hyperparams"]

    df_raw = load_results()
    df, final_ratings = compute_elo(df_raw, home_advantage=ha)
    df = compute_h2h(df, p["global_home_avg"], p["global_away_avg"], k=hp["h2h_k"])
    df = add_tournament_features(df)

    team_list = sorted(final_ratings.keys())
    wc_form, wc_avg_gpg                   = get_wc_form(df)
    wc_form_v2, wc_avg_qa_gpg, wc_avg_qa_cpg = get_wc_form_v2(df, final_ratings)
    fifa_data = load_latest_fifa_data(FIFA_DATA_DIR)
    return model, cfg, df, final_ratings, team_list, wc_form, wc_avg_gpg, wc_form_v2, wc_avg_qa_gpg, wc_avg_qa_cpg, fifa_data


@st.cache_data
def load_wc_teams() -> list[str]:
    df = pd.read_csv("international_results-master/mundial2026.csv")
    return sorted(set(df["home_team"]) | set(df["away_team"]))


(model, cfg, df_full, final_ratings, team_list,
 wc_form, wc_avg_gpg,
 wc_form_v2, wc_avg_qa_gpg, wc_avg_qa_cpg,
 fifa_data) = load_pipeline()

WC_TEAMS = [t for t in load_wc_teams() if t in final_ratings]


# ── prediction helper ─────────────────────────────────────────────────────────
def predict(
    home_team: str,
    away_team: str,
    neutral: bool,
    knockout: bool,
    with_form: bool = False,
    form_alpha: float = 0.7,
    fifa_data: dict = None,
) -> dict:
    p = cfg["priors"]
    k = cfg["hyperparams"]["h2h_k"]

    r_h = final_ratings[home_team]
    r_a = final_ratings[away_team]

    # Normalize knockout neutral matches to higher-Elo team as "home"
    # for deterministic H2H lookup, then unswap probabilities before returning
    # so the UI always shows values aligned to the user's original selectbox order.
    swapped = False
    if knockout and neutral and r_a > r_h:
        home_team, away_team = away_team, home_team
        r_h, r_a = r_a, r_h
        swapped = True

    mask  = (df_full["home_team"] == home_team) & (df_full["away_team"] == away_team)
    hist  = df_full[mask]
    n_h2h = len(hist)

    if n_h2h == 0:
        h2h_home = p["global_home_avg"]
        h2h_away = p["global_away_avg"]
    else:
        h2h_home = (hist["home_score"].sum() + k * p["global_home_avg"]) / (n_h2h + k)
        h2h_away = (hist["away_score"].sum() + k * p["global_away_avg"]) / (n_h2h + k)

    row = pd.DataFrame([{
        "elo_diff":          r_h - r_a,
        "neutral":           float(neutral),
        "h2h_home_goals_mu": h2h_home,
        "h2h_away_goals_mu": h2h_away,
        "is_world_cup":      1.0,
        "is_knockout":       1.0 if knockout else 0.0,
    }])

    theta_D         = cfg["theta_D_knockout"] if knockout else cfg["theta_D"]
    lam_h_arr, lam_a_arr = model.predict_lambdas(row)
    proba           = model.predict_proba(row)
    scoreline       = model.predict_scoreline(row)

    p_h, p_d, p_a = float(proba[0, 0]), float(proba[0, 1]), float(proba[0, 2])
    sh = int(scoreline["pred_home"].iloc[0])
    sa = int(scoreline["pred_away"].iloc[0])

    if p_d > theta_D:
        prediction = "Draw"
    elif p_h >= p_a:
        prediction = home_team
    else:
        prediction = away_team

    # ── form adjustment ───────────────────────────────────────────────────────
    form_info = None
    form_info_v2 = None
    if with_form:
        lam_h_adj, lam_a_adj = form_adjusted_lambdas(
            float(lam_h_arr[0]), float(lam_a_arr[0]),
            home_team, away_team,
            wc_form, wc_avg_gpg, alpha=form_alpha,
        )
        proba_adj, sh_adj_arr, sa_adj_arr = model.predict_from_lambdas(
            np.array([lam_h_adj]), np.array([lam_a_adj])
        )
        ph_adj = float(proba_adj[0, 0])
        pd_adj = float(proba_adj[0, 1])
        pa_adj = float(proba_adj[0, 2])
        sh_adj = int(sh_adj_arr[0])
        sa_adj = int(sa_adj_arr[0])
        def _fs(team):
            s = wc_form.get(team, {})
            return {
                "games":    s.get("games", 0),
                "scored":   s.get("scored", 0),
                "conceded": s.get("conceded", 0),
                "gd":       s.get("gd", 0),
                "gpg":      s.get("gpg", 0.0),
            }

        # Original (post-main-unswap) team names
        original_h = away_team if swapped else home_team
        original_a = home_team if swapped else away_team

        # Unswap form probs and scores to align with original team order
        if swapped:
            ph_adj, pa_adj = pa_adj, ph_adj
            sh_adj, sa_adj = sa_adj, sh_adj

        # Compute prediction label using unswapped probs and original names
        if pd_adj > theta_D:
            pred_adj = "Draw"
        elif ph_adj >= pa_adj:
            pred_adj = original_h
        else:
            pred_adj = original_a

        form_info = {
            "p_h": ph_adj, "p_d": pd_adj, "p_a": pa_adj,
            "score_h": sh_adj, "score_a": sa_adj,
            "prediction": pred_adj,
            "home_form": _fs(original_h),
            "away_form": _fs(original_a),
            "avg_gpg": wc_avg_gpg,
            "alpha": form_alpha,
        }

        # ── form v2: attack + defense + opponent quality ───────────────────
        lam_h_v2, lam_a_v2 = form_adjusted_lambdas_v2(
            float(lam_h_arr[0]), float(lam_a_arr[0]),
            home_team, away_team,
            wc_form_v2, wc_avg_qa_gpg, wc_avg_qa_cpg,
            alpha=form_alpha,
        )
        proba_v2, sh_v2_arr, sa_v2_arr = model.predict_from_lambdas(
            np.array([lam_h_v2]), np.array([lam_a_v2])
        )
        ph_v2 = float(proba_v2[0, 0])
        pd_v2 = float(proba_v2[0, 1])
        pa_v2 = float(proba_v2[0, 2])
        sh_v2 = int(sh_v2_arr[0])
        sa_v2 = int(sa_v2_arr[0])

        if swapped:
            ph_v2, pa_v2 = pa_v2, ph_v2
            sh_v2, sa_v2 = sa_v2, sh_v2

        if pd_v2 > theta_D:
            pred_v2 = "Draw"
        elif ph_v2 >= pa_v2:
            pred_v2 = original_h
        else:
            pred_v2 = original_a

        def _fs_v2(team):
            s = wc_form_v2.get(team, {})
            return {"games": s.get("games", 0), "qa_gpg": s.get("qa_gpg", 0.0), "qa_cpg": s.get("qa_cpg", 0.0)}

        # Top scorelines for form-adjusted prediction
        from scipy.stats import poisson as _pois
        _G = 8
        _g = np.arange(_G)
        _pm = (_pois.pmf(_g[:, None], lam_h_v2) *
               _pois.pmf(_g[None, :], lam_a_v2))
        _rho = model.rho_
        _pm[0, 0] *= max(1 - lam_h_v2 * lam_a_v2 * _rho, 0)
        _pm[0, 1] *= max(1 + lam_h_v2 * _rho, 0)
        _pm[1, 0] *= max(1 + lam_a_v2 * _rho, 0)
        _pm[1, 1] *= max(1 - _rho, 0)
        _top_v2 = sorted(
            [(_pm[i, j], i, j) for i in range(_G) for j in range(_G)],
            reverse=True,
        )[:8]
        if swapped:
            _top_v2 = [(p, j, i) for p, i, j in _top_v2]

        form_info_v2 = {
            "p_h": ph_v2, "p_d": pd_v2, "p_a": pa_v2,
            "score_h": sh_v2, "score_a": sa_v2,
            "prediction": pred_v2,
            "home_form": _fs_v2(original_h),
            "away_form": _fs_v2(original_a),
            "avg_qa_gpg": wc_avg_qa_gpg,
            "avg_qa_cpg": wc_avg_qa_cpg,
            "alpha": form_alpha,
            "top_scores": _top_v2,
        }

        # ── form v3: form_v2 + FIFA advanced stats ─────────────────────
        form_info_v3 = None
        if fifa_data:
            lam_h_v3, lam_a_v3 = form_adjusted_lambdas_v3(
                float(lam_h_arr[0]), float(lam_a_arr[0]),
                home_team, away_team,
                wc_form_v2, wc_avg_qa_gpg, wc_avg_qa_cpg,
                fifa_data,
                alpha_form=form_alpha,
                alpha_fifa=0.90,
            )
            proba_v3, sh_v3_arr, sa_v3_arr = model.predict_from_lambdas(
                np.array([lam_h_v3]), np.array([lam_a_v3])
            )
            ph_v3 = float(proba_v3[0, 0])
            pd_v3 = float(proba_v3[0, 1])
            pa_v3 = float(proba_v3[0, 2])
            sh_v3 = int(sh_v3_arr[0])
            sa_v3 = int(sa_v3_arr[0])

            if swapped:
                ph_v3, pa_v3 = pa_v3, ph_v3
                sh_v3, sa_v3 = sa_v3, sh_v3

            if pd_v3 > theta_D:
                pred_v3 = "Draw"
            elif ph_v3 >= pa_v3:
                pred_v3 = original_h
            else:
                pred_v3 = original_a

            _pm3 = (_pois.pmf(_g[:, None], lam_h_v3) *
                    _pois.pmf(_g[None, :], lam_a_v3))
            _pm3[0, 0] *= max(1 - lam_h_v3 * lam_a_v3 * _rho, 0)
            _pm3[0, 1] *= max(1 + lam_h_v3 * _rho, 0)
            _pm3[1, 0] *= max(1 + lam_a_v3 * _rho, 0)
            _pm3[1, 1] *= max(1 - _rho, 0)
            _top_v3 = sorted(
                [(_pm3[i, j], i, j) for i in range(_G) for j in range(_G)],
                reverse=True,
            )[:8]
            if swapped:
                _top_v3 = [(p, j, i) for p, i, j in _top_v3]

            _neutral_fifa = {"attack_factor": 1.0, "defense_factor": 1.0}
            form_info_v3 = {
                "p_h": ph_v3, "p_d": pd_v3, "p_a": pa_v3,
                "score_h": sh_v3, "score_a": sa_v3,
                "prediction": pred_v3,
                "home_form": _fs_v2(original_h),
                "away_form": _fs_v2(original_a),
                "home_fifa": fifa_data.get(original_h, _neutral_fifa),
                "away_fifa": fifa_data.get(original_a, _neutral_fifa),
                "avg_qa_gpg": wc_avg_qa_gpg,
                "avg_qa_cpg": wc_avg_qa_cpg,
                "alpha": form_alpha,
                "top_scores": _top_v3,
            }

    # Unswap so probabilities align with the user's original selection
    if swapped:
        p_h, p_a = p_a, p_h
        sh, sa   = sa, sh
        r_h, r_a = r_a, r_h
        home_team, away_team = away_team, home_team

    # Top scorelines from the joint Poisson matrix (computed before unswap)
    from scipy.stats import poisson as poisson_dist
    MAX_G = 8
    goals = np.arange(MAX_G)
    lam_h_arr, lam_a_arr = model.predict_lambdas(row)
    lam_h_val, lam_a_val = float(lam_h_arr[0]), float(lam_a_arr[0])
    p_mat = (poisson_dist.pmf(goals[:, None], lam_h_val) *
             poisson_dist.pmf(goals[None, :], lam_a_val))
    rho = model.rho_
    p_mat[0, 0] *= max(1 - lam_h_val * lam_a_val * rho, 0)
    p_mat[0, 1] *= max(1 + lam_h_val * rho, 0)
    p_mat[1, 0] *= max(1 + lam_a_val * rho, 0)
    p_mat[1, 1] *= max(1 - rho, 0)
    flat = [(p_mat[i, j], i, j) for i in range(MAX_G) for j in range(MAX_G)]
    top  = sorted(flat, reverse=True)[:8]
    # If teams were swapped for H2H, flip home/away goals to match original order
    if swapped:
        top = [(prob, j, i) for prob, i, j in top]

    return {
        "p_h": p_h, "p_d": p_d, "p_a": p_a,
        "score_h": sh, "score_a": sa,
        "elo_h": round(r_h), "elo_a": round(r_a),
        "elo_diff": round(r_h - r_a),
        "h2h_n": n_h2h,
        "prediction": prediction,
        "theta_D": theta_D,
        "top_scores": top,
        "form": form_info,
        "form_v2": form_info_v2,
        "form_v3": form_info_v3 if with_form else None,
    }


# ── UI helpers ────────────────────────────────────────────────────────────────

def _prob_bars(home: str, away: str, p_h: float, p_d: float, p_a: float) -> None:
    rows = [
        (f"{flag(home)} {home}", p_h, "#2ecc71"),
        ("🤝 Draw",              p_d, "#f0a500"),
        (f"{flag(away)} {away}", p_a, "#e74c3c"),
    ]
    html = "<div style='margin:6px 0 16px 0'>"
    for label, prob, color in rows:
        pct = prob * 100
        html += (
            f"<div style='margin-bottom:10px'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:3px'>"
            f"<span style='font-size:0.9em'>{label}</span>"
            f"<span style='font-size:0.9em;font-weight:700'>{prob:.1%}</span>"
            f"</div>"
            f"<div style='background:#e8e8e8;border-radius:6px;height:13px;overflow:hidden'>"
            f"<div style='background:{color};border-radius:6px;height:13px;width:{pct:.1f}%'></div>"
            f"</div></div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _outcome_card(prediction: str, home_team: str, away_team: str,
                  p_h: float, p_d: float, p_a: float, theta_D: float) -> None:
    if prediction == "Draw":
        st.markdown(
            f"<div style='background:#fff8e1;border-left:5px solid #f0a500;"
            f"padding:14px 18px;border-radius:6px;margin:4px 0'>"
            f"<span style='font-size:1.3em'>🤝 <strong>Draw</strong></span><br>"
            f"<span style='color:#666;font-size:0.9em'>"
            f"P(Draw) = {p_d:.1%} — above threshold {theta_D:.2f}</span></div>",
            unsafe_allow_html=True,
        )
    else:
        winner_prob = p_h if prediction == home_team else p_a
        st.markdown(
            f"<div style='background:#e8f5e9;border-left:5px solid #2ecc71;"
            f"padding:14px 18px;border-radius:6px;margin:4px 0'>"
            f"<span style='font-size:1.3em'>{flag(prediction)} "
            f"<strong>{prediction} wins</strong></span><br>"
            f"<span style='color:#666;font-size:0.9em'>"
            f"{winner_prob:.1%} win probability</span></div>",
            unsafe_allow_html=True,
        )


def _scoreline_display(home: str, away: str, score_h: int, score_a: int,
                       heading_level: str = "h2") -> None:
    c1, c2, c3 = st.columns([3, 2, 3])
    with c1:
        st.markdown(
            f"<{heading_level} style='text-align:right;margin:0'>"
            f"{flag(home)} {home}</{heading_level}>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<{heading_level} style='text-align:center;margin:0'>"
            f"{score_h} – {score_a}</{heading_level}>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<{heading_level} style='text-align:left;margin:0'>"
            f"{away} {flag(away)}</{heading_level}>",
            unsafe_allow_html=True,
        )


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("⚽ WC 2026 Predictor")
st.caption(
    "Model v5 · Elo + H2H + WC phase features · "
    "Ratings updated through current group stage"
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox(
        "Team 1",
        options=WC_TEAMS,
        format_func=lambda t: f"{FLAGS.get(t, '🏳️')} {t}",
        index=WC_TEAMS.index("Argentina") if "Argentina" in WC_TEAMS else 0,
        key="home",
    )
with col2:
    away_team = st.selectbox(
        "Team 2",
        options=WC_TEAMS,
        format_func=lambda t: f"{FLAGS.get(t, '🏳️')} {t}",
        index=WC_TEAMS.index("France") if "France" in WC_TEAMS else 1,
        key="away",
    )

col3, col4 = st.columns(2)
with col3:
    neutral = st.checkbox("Neutral venue", value=True)
with col4:
    knockout = st.checkbox(
        "Knockout phase",
        value=False,
        help=f"Uses theta_D_knockout={cfg['theta_D_knockout']:.2f} instead of {cfg['theta_D']:.2f}",
    )

with_form = st.checkbox(
    "Tournament form adjustment",
    value=False,
    help=(
        f"Blends v5 prediction with each team's WC 2026 attack form (goals/game). "
        f"α=0.70 → 70% model + 30% form. "
        f"Tournament avg: {wc_avg_gpg:.2f} gpg/team ({len(wc_form)} teams with data)."
    ),
)

predict_btn = st.button("Predict", type="primary", use_container_width=True)

# ── result ────────────────────────────────────────────────────────────────────
if predict_btn:
    if home_team == away_team:
        st.warning("Select two different teams.")
    else:
        r = predict(home_team, away_team, neutral, knockout, with_form=with_form, fifa_data=fifa_data)

        st.divider()

        phase_label = "Knockout" if knockout else "Group stage"
        st.caption(f"Phase: {phase_label} · Elo: {flag(home_team)} {home_team} {r['elo_h']} vs {flag(away_team)} {away_team} {r['elo_a']} (diff {r['elo_diff']:+}) · θ_D = {r['theta_D']:.2f}")

        # ── BLOCK 1: Predicted outcome ────────────────────────────────────────
        st.subheader("Predicted outcome")
        _outcome_card(
            r["prediction"], home_team, away_team,
            r["p_h"], r["p_d"], r["p_a"], r["theta_D"],
        )

        st.subheader("Win probabilities")
        _prob_bars(home_team, away_team, r["p_h"], r["p_d"], r["p_a"])

        # ── BLOCK 2: Most probable scoreline ─────────────────────────────────
        st.divider()
        st.subheader("Most probable scoreline")

        score_is_draw = r["score_h"] == r["score_a"]
        pred_is_draw  = r["prediction"] == "Draw"

        if score_is_draw and not pred_is_draw:
            winner      = r["prediction"]
            winner_prob = r["p_h"] if r["prediction"] == home_team else r["p_a"]
            st.caption(
                f"The most likely individual scoreline is a draw, but **{winner}** "
                f"wins more often overall ({winner_prob:.1%}) by accumulating probability "
                f"across many winning scorelines combined (e.g. 1-0, 2-0, 2-1…). "
                f"The draw probability ({r['p_d']:.1%}) is below the decision threshold ({r['theta_D']:.2f})."
            )
        elif not score_is_draw and pred_is_draw:
            st.caption(
                f"The most likely individual scoreline is not a draw, but the combined "
                f"draw probability ({r['p_d']:.1%}) exceeds the threshold ({r['theta_D']:.2f}), "
                f"so the model predicts a draw."
            )

        _scoreline_display(home_team, away_team, r["score_h"], r["score_a"])

        # ── BLOCK 3: Scoreline distribution ──────────────────────────────────
        st.divider()
        st.subheader("Scoreline distribution")
        st.caption(
            "Most probable individual scorelines — not calibrated predictions. "
            "The outcome above is what the model is calibrated for."
        )

        top = r["top_scores"]
        cols = st.columns(4)
        for idx, (prob, gh, ga) in enumerate(top[:8]):
            outcome = (
                home_team.split()[0] if gh > ga else
                away_team.split()[0] if ga > gh else
                "Draw"
            )
            cols[idx % 4].metric(
                label=f"{gh} – {ga}",
                value=f"{prob:.1%}",
                delta=outcome,
                delta_color="off",
            )

        # ── BLOCK 4: Tournament Form Adjustment (Experimental) ────────────────
        f_main = r.get("form_v3") or r.get("form_v2")
        if f_main:
            f2 = f_main
            f1 = r["form"]
            has_fifa = r.get("form_v3") is not None

            st.divider()

            st.markdown(
                "### Tournament Form Adjustment &nbsp;"
                "<span style='background:#f0a500;color:#fff;padding:2px 8px;"
                "border-radius:4px;font-size:0.7em;font-weight:bold;"
                "vertical-align:middle'>EXPERIMENTAL</span>",
                unsafe_allow_html=True,
            )
            fifa_note = " + FIFA advanced stats" if has_fifa else ""
            st.caption(
                f"v5 base model blended with WC 2026 in-tournament performance{fifa_note} · "
                f"alpha={f2['alpha']:.2f}  (70% v5 + 30% form signal)"
            )

            with st.expander("What does this add to v5?"):
                fifa_extra = (
                    "\n\n**FIFA advanced stats layer (×0.90 blend on top of form):**\n"
                    "- **xG + efficiency** — expected goals and over/underperformance\n"
                    "- **Goalkeeper saves** — direct keeper quality signal\n"
                    "- **Defensive pressures & recovery time** — defensive intensity\n"
                    "- **Runs behind defense + receptions under pressure** — attack movement\n"
                    "- **Possession % + avg speed** — game control and physical intensity\n"
                ) if has_fifa else ""
                st.markdown(
                    "**v5 features:** Elo difference · head-to-head history · "
                    "neutral venue · WC / knockout phase\n\n"
                    "**Form layer adds (quality-adjusted):**\n"
                    "- **Attack** — goals scored per game in WC 2026, weighted by "
                    "opponent Elo (goals vs stronger teams count more)\n"
                    "- **Defense** — goals conceded per game, weighted by opponent Elo "
                    "(conceding vs weaker teams is penalized more)\n\n"
                    "**Blend formula:**\n"
                    "```\n"
                    "lambda_adj = lambda_v5 × (0.70 + 0.30 × sqrt(attack_ratio × opp_def_ratio))\n"
                    "```"
                    + fifa_extra +
                    "\n\nTeams with no WC 2026 data receive no adjustment (ratio = 1.0)."
                )

            # ── Team form stats ───────────────────────────────────────────────
            st.markdown("**WC 2026 performance**")
            fc1, fc2 = st.columns(2)

            def _form_card(col, team, s1, s2, avg_qa_gpg, avg_qa_cpg, fifa_info=None):
                with col:
                    g = s1.get("games", 0)
                    if g == 0:
                        st.info(f"{flag(team)} **{team}**  \nNo WC 2026 data yet")
                        return
                    qa_atk = s2["qa_gpg"] / avg_qa_gpg if avg_qa_gpg > 0 else 1.0
                    qa_def = s2["qa_cpg"] / avg_qa_cpg if avg_qa_cpg > 0 else 1.0
                    atk_label = "strong" if qa_atk > 1.10 else ("poor" if qa_atk < 0.90 else "avg")
                    def_label = "solid" if qa_def < 0.90 else ("leaky" if qa_def > 1.10 else "avg")
                    body = (
                        f"{flag(team)} **{team}**  \n"
                        f"GP {g} &nbsp;·&nbsp; GF {s1['scored']} &nbsp;·&nbsp; "
                        f"GA {s1['conceded']} &nbsp;·&nbsp; GD {s1['gd']:+}  \n"
                        f"Attack: {s2['qa_gpg']:.2f} qa-gpg &nbsp;({qa_atk:.2f}x avg · *{atk_label}*)  \n"
                        f"Defense: {s2['qa_cpg']:.2f} qa-cpg &nbsp;({qa_def:.2f}x avg · *{def_label}*)"
                    )
                    if fifa_info:
                        atk_f = fifa_info["attack_factor"]
                        def_f = fifa_info["defense_factor"]
                        atk_fl = "↑" if atk_f > 1.05 else ("↓" if atk_f < 0.95 else "=")
                        def_fl = "↑" if def_f > 1.05 else ("↓" if def_f < 0.95 else "=")
                        body += (
                            f"  \nFIFA · Attack {atk_f:.2f}x {atk_fl} "
                            f"&nbsp;·&nbsp; Defense {def_f:.2f}x {def_fl}"
                        )
                    st.markdown(body, unsafe_allow_html=True)

            home_fifa_info = f2.get("home_fifa") if has_fifa else None
            away_fifa_info = f2.get("away_fifa") if has_fifa else None

            _form_card(fc1, home_team, f1["home_form"], f2["home_form"],
                       f2["avg_qa_gpg"], f2["avg_qa_cpg"], home_fifa_info)
            _form_card(fc2, away_team, f1["away_form"], f2["away_form"],
                       f2["avg_qa_gpg"], f2["avg_qa_cpg"], away_fifa_info)

            # ── Predicted outcome ─────────────────────────────────────────────
            st.markdown("**Predicted outcome — form adjusted**")
            pred_changed = f2["prediction"] != r["prediction"]
            change_note  = f"  *(changed from v5: {r['prediction']})*" if pred_changed else ""

            _outcome_card(
                f2["prediction"], home_team, away_team,
                f2["p_h"], f2["p_d"], f2["p_a"], r["theta_D"],
            )
            if change_note:
                st.caption(f"Prediction changed vs v5: was **{r['prediction']}**")
            else:
                st.caption(f"Same prediction as v5 — form signal confirms {r['prediction']}")

            # ── Probability comparison ─────────────────────────────────────────
            st.markdown("**Win probabilities — v5 vs form-adjusted**")
            mc1, mc2, mc3 = st.columns(3)
            for col, lbl, v5v, fv in zip(
                [mc1, mc2, mc3],
                [home_team, "Draw", away_team],
                [r["p_h"], r["p_d"], r["p_a"]],
                [f2["p_h"], f2["p_d"], f2["p_a"]],
            ):
                with col:
                    st.metric(
                        label=f"{flag(lbl)} {lbl}" if lbl != "Draw" else "🤝 Draw",
                        value=f"{fv:.1%}",
                        delta=f"{fv - v5v:+.1%} vs v5",
                        help=f"v5: {v5v:.1%}  →  form-adjusted: {fv:.1%}",
                    )

            st.markdown("v5 base")
            _prob_bars(home_team, away_team, r["p_h"], r["p_d"], r["p_a"])
            st.markdown("Form-adjusted")
            _prob_bars(home_team, away_team, f2["p_h"], f2["p_d"], f2["p_a"])

            # ── Most probable scoreline ────────────────────────────────────────
            st.markdown("**Most probable scoreline — form adjusted**")
            _scoreline_display(home_team, away_team, f2["score_h"], f2["score_a"], "h3")

            # ── Scoreline distribution ─────────────────────────────────────────
            st.markdown("**Scoreline distribution — form adjusted**")
            scols = st.columns(4)
            for idx, (prob, gh, ga) in enumerate(f2["top_scores"][:8]):
                outcome = (
                    home_team.split()[0] if gh > ga else
                    away_team.split()[0] if ga > gh else
                    "Draw"
                )
                scols[idx % 4].metric(
                    label=f"{gh} – {ga}",
                    value=f"{prob:.1%}",
                    delta=outcome,
                    delta_color="off",
                )

        # ── Details ───────────────────────────────────────────────────────────
        with st.expander("Details"):
            st.write(
                f"**Elo:** {flag(home_team)} {home_team} {r['elo_h']} · "
                f"{flag(away_team)} {away_team} {r['elo_a']} (diff {r['elo_diff']:+})"
            )
            st.write(f"**Head-to-head:** {r['h2h_n']} previous matches in this direction")
            fav = home_team if r["p_h"] > r["p_a"] else away_team if r["p_a"] > r["p_h"] else "Even"
            st.write(f"**Elo favourite:** {flag(fav)} {fav}" if fav != "Even" else "**Elo favourite:** Even")
