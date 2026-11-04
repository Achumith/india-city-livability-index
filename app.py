"""
God-Level Interactive Web Application
India City Livability Index & Karnataka State Deep-Dive
Author: Ruchira Ramesh Yallur | BMSITM, Bengaluru
"""

import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.calculator import (
    PERSONA_PRESETS,
    PILLAR_TO_RAW_COL,
    PILLAR_TO_SCORE_COL,
    SCORE_COL_TO_PILLAR,
    calculate_livability_index,
    compare_two_entities,
    find_best_match_cities,
    generate_city_narrative,
    get_top_ranked,
)
from src.ml_engine import run_full_ml_pipeline
from src.plots import (
    create_comparison_radar_chart,
    create_diverging_diff_bar,
    create_india_map,
    create_karnataka_map,
    create_leaderboard_bar,
    create_pca_cluster_scatter,
    create_radar_chart,
)
from src.preprocessor import SCORE_COLUMNS, generate_master_datasets

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & ELITE STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="India & Karnataka Livability Index • Urban Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Universal Theme & Contrast Enforcement */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* Force all general text elements to dark slate */
    h1, h2, h3, h4, h5, h6, p, span, label, li, [data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: #475569 !important;
    }

    /* Streamlit Metrics */
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        color: #475569 !important;
        font-weight: 600 !important;
    }

    /* Tabs styling */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        color: #475569 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0284c7 !important;
        border-bottom-color: #0284c7 !important;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #091e42 0%, #0f2b5c 50%, #0284c7 100%) !important;
        border-radius: 16px !important;
        padding: 28px 32px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.25) !important;
    }
    .hero-container, .hero-container * {
        color: #ffffff !important;
    }
    .hero-badge {
        display: inline-block !important;
        padding: 4px 12px !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 20px !important;
        margin-bottom: 12px !important;
        color: #ffffff !important;
    }
    .hero-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        line-height: 1.15 !important;
        margin-bottom: 8px !important;
        color: #ffffff !important;
    }
    .hero-subtitle {
        font-size: 1.05rem !important;
        color: #e2e8f0 !important;
        max-width: 850px !important;
        line-height: 1.5 !important;
    }

    /* KPI Cards */
    .kpi-card {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 2px 6px -1px rgba(0, 0, 0, 0.04) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px -2px rgba(0, 0, 0, 0.08) !important;
    }
    .kpi-card, .kpi-card * {
        color: #0f172a !important;
    }
    .kpi-card .kpi-label {
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #64748b !important;
        margin-bottom: 4px !important;
    }
    .kpi-card .kpi-value {
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        line-height: 1.1 !important;
    }
    .kpi-card .kpi-sub, .kpi-card .kpi-sub * {
        font-size: 0.82rem !important;
        color: #0284c7 !important;
        font-weight: 600 !important;
        margin-top: 4px !important;
    }

    /* Matchmaker Card */
    .match-card {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 14px !important;
        padding: 20px 22px !important;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 16px !important;
    }
    .match-card, .match-card * {
        color: #0f172a !important;
    }
    .match-card .match-meta {
        font-size: 0.82rem !important;
        color: #64748b !important;
        margin-bottom: 12px !important;
    }
    .match-card .match-score {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #0284c7 !important;
        margin-bottom: 12px !important;
    }
    .match-card .match-archetype {
        font-size: 0.8rem !important;
        background: #f1f5f9 !important;
        padding: 6px 10px !important;
        border-radius: 6px !important;
        color: #334155 !important;
        border: 1px solid #e2e8f0 !important;
    }
    .match-ring {
        display: inline-block !important;
        padding: 6px 14px !important;
        border-radius: 20px !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        background: #dbeafe !important;
        color: #1d4ed8 !important;
    }

    /* Head-to-Head Arena Card */
    .arena-card {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 14px !important;
        padding: 18px 24px !important;
        margin-bottom: 20px !important;
        display: flex !important;
        justify-content: space-around !important;
        align-items: center !important;
        text-align: center !important;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05) !important;
    }
    .arena-card, .arena-card * {
        color: #0f172a !important;
    }
    .arena-name-a {
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        color: #0284c7 !important;
    }
    .arena-score-a {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }
    .arena-sub-a {
        font-size: 0.82rem !important;
        color: #475569 !important;
    }
    .arena-vs {
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        color: #94a3b8 !important;
    }
    .arena-name-b {
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        color: #ea580c !important;
    }
    .arena-score-b {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }
    .arena-sub-b {
        font-size: 0.82rem !important;
        color: #475569 !important;
    }

    /* 360 Dossier Card */
    .dossier-card {
        background: #f0fdf4 !important;
        border: 1px solid #bbf7d0 !important;
        border-left: 5px solid #10b981 !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        margin-bottom: 22px !important;
        box-shadow: 0 4px 12px -2px rgba(16, 185, 129, 0.08) !important;
    }
    .dossier-card, .dossier-card * {
        color: #0f172a !important;
    }
    .dossier-title {
        font-weight: 800 !important;
        color: #15803d !important;
        font-size: 1.15rem !important;
        margin-bottom: 6px !important;
    }
    .dossier-summary {
        color: #1e293b !important;
        font-size: 0.95rem !important;
        line-height: 1.55 !important;
        margin-bottom: 12px !important;
    }

    /* Pillar badges */
    .badge-superpower {
        background-color: #dcfce7 !important;
        color: #15803d !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        display: inline-block !important;
    }
    .badge-bottleneck {
        background-color: #fee2e2 !important;
        color: #b91c1c !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        display: inline-block !important;
    }

    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] .stCaption {
        color: #64748b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. DATA LOADING & CACHING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_data():
    nat_csv = PROJECT_ROOT / "data" / "processed" / "master_national.csv"
    ka_csv = PROJECT_ROOT / "data" / "processed" / "master_karnataka.csv"

    if not nat_csv.exists() or not ka_csv.exists():
        master_nat, master_ka = generate_master_datasets()
    else:
        master_nat = pd.read_csv(nat_csv)
        master_ka = pd.read_csv(ka_csv)

    # Pre-train and cache ML models
    enriched_nat, ml_nat = run_full_ml_pipeline(master_nat, "Pan-India", n_clusters=4)
    enriched_ka, ml_ka = run_full_ml_pipeline(master_ka, "Karnataka", n_clusters=4)

    return {
        "raw_nat": master_nat,
        "raw_ka": master_ka,
        "enriched_nat": enriched_nat,
        "enriched_ka": enriched_ka,
        "ml_nat": ml_nat,
        "ml_ka": ml_ka,
    }


data_bundle = load_all_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR: SCOPE, PRESETS & SLIDERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Geographic Scope")
    view_mode = st.radio(
        "Select Exploration Level:",
        options=["Pan-India (30 Major Metros)", "Karnataka State (31 Districts)"],
        index=0,
    )
    is_national = "Pan-India" in view_mode
    active_dataset = data_bundle["enriched_nat"] if is_national else data_bundle["enriched_ka"]
    active_ml = data_bundle["ml_nat"] if is_national else data_bundle["ml_ka"]
    tier_label = "National" if is_national else "Karnataka"

    st.markdown("---")
    st.markdown("### Priority Profile")
    persona_options = list(PERSONA_PRESETS.keys()) + ["Custom"]

    if "current_persona" not in st.session_state:
        st.session_state.current_persona = "Balanced"
    if "slider_weights" not in st.session_state:
        st.session_state.slider_weights = PERSONA_PRESETS["Balanced"].copy()

    def on_persona_change():
        p = st.session_state.persona_select
        if p != "Custom":
            st.session_state.slider_weights = PERSONA_PRESETS[p].copy()
            st.session_state.current_persona = p

    selected_persona = st.selectbox(
        "Preset Configuration:",
        options=persona_options,
        index=persona_options.index(st.session_state.current_persona),
        key="persona_select",
        on_change=on_persona_change,
    )

    # Preset quick description
    persona_descriptions = {
        "Balanced": "Equal weighting (5/10) across all 7 livability dimensions.",
        "Student": "Prioritizes premier education institutions and affordable living costs.",
        "Family": "Emphasizes public safety, healthcare infrastructure, and clean air quality.",
        "Retiree": "Focuses on healthcare accessibility, mild climate comfort, and safety.",
        "Remote Worker": "Emphasizes high-speed digital connectivity and affordable housing.",
        "Job Seeker": "Prioritizes economic transit infrastructure and growth opportunities.",
        "Custom": "User-defined weight distribution across dimensions.",
    }
    st.caption(persona_descriptions.get(selected_persona, ""))

    col_rst1, col_rst2 = st.columns(2)
    with col_rst1:
        if st.button("Auto-Balance (5.0)", width="stretch"):
            st.session_state.slider_weights = {p: 5.0 for p in PERSONA_PRESETS["Balanced"]}
            st.session_state.current_persona = "Balanced"
            st.rerun()
    with col_rst2:
        if st.button("Reset Defaults", width="stretch"):
            st.session_state.slider_weights = PERSONA_PRESETS["Balanced"].copy()
            st.session_state.current_persona = "Balanced"
            st.rerun()

    st.markdown("---")
    st.markdown("### Livability Dimension Weights (1–10)")
    st.caption("Adjust to simulate personal priorities in real-time:")

    user_weights: Dict[str, float] = {}
    pillars = [
        "Climate",
        "Air Quality",
        "Safety",
        "Education",
        "Affordability",
        "Healthcare",
        "Connectivity",
    ]

    for pillar in pillars:
        curr_val = float(st.session_state.slider_weights.get(pillar, 5.0))
        val = st.slider(
            pillar,
            min_value=1.0,
            max_value=10.0,
            value=curr_val,
            step=1.0,
            key=f"slider_{pillar}",
        )
        user_weights[pillar] = val
        if val != curr_val and st.session_state.current_persona != "Custom":
            st.session_state.current_persona = "Custom"
        st.session_state.slider_weights[pillar] = val

    st.markdown("---")
    st.markdown("#### Filter Sub-Region")
    subregions = sorted(active_dataset["region"].unique().tolist()) if "region" in active_dataset.columns else []
    selected_subregion = st.selectbox("Sub-Region Filter:", options=["All Sub-Regions"] + subregions)

    st.markdown("---")
    st.caption("MCA Project | **Ruchira Ramesh Yallur** (BMSITM)")

# -----------------------------------------------------------------------------
# 4. CALCULATE DYNAMIC WEIGHTED LIVABILITY SCORES
# -----------------------------------------------------------------------------
ranked_df = calculate_livability_index(active_dataset, weights=user_weights)

# Apply optional sub-region filter
if selected_subregion != "All Sub-Regions" and "region" in ranked_df.columns:
    display_df = ranked_df[ranked_df["region"] == selected_subregion].copy()
else:
    display_df = ranked_df.copy()

# -----------------------------------------------------------------------------
# 5. HERO HEADER & KPI STAT RIBBON
# -----------------------------------------------------------------------------
scope_title = "Pan-India Metropolitan Benchmark" if is_national else "Karnataka State District Deep-Dive"
scope_desc = (
    "Evaluating 30 major Indian urban centers across 7 fundamental dimensions of urban quality-of-life."
    if is_national
    else "Granular socio-economic and environmental livability assessment across all 31 districts of Karnataka."
)

st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-badge">Urban Intelligence Platform • 2026</div>
        <div class="hero-title">{scope_title}</div>
        <div class="hero-subtitle">{scope_desc}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Metric KPI Ribbon
top_entity = ranked_df.iloc[0]
runner_up_entity = ranked_df.iloc[1] if len(ranked_df) > 1 else top_entity
avg_livability = ranked_df["custom_score"].mean()
total_weight = int(sum(user_weights.values()))

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #475569 !important;">Rank #1 Entity</div>
            <div class="kpi-value" style="color: #0f172a !important;">{top_entity['name']}</div>
            <div class="kpi-sub" style="color: #0284c7 !important;">Score: <b style="color: #0284c7 !important;">{top_entity['custom_score']:.1f}</b>/100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #475569 !important;">Rank #2 Runner-Up</div>
            <div class="kpi-value" style="color: #0f172a !important;">{runner_up_entity['name']}</div>
            <div class="kpi-sub" style="color: #0284c7 !important;">Score: <b style="color: #0284c7 !important;">{runner_up_entity['custom_score']:.1f}</b>/100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #475569 !important;">Overall Mean Score</div>
            <div class="kpi-value" style="color: #0f172a !important;">{avg_livability:.1f}</div>
            <div class="kpi-sub" style="color: #0284c7 !important;">{len(ranked_df)} Total Evaluated</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #475569 !important;">Active Weight Profile</div>
            <div class="kpi-value" style="color: #0f172a !important;">{st.session_state.current_persona}</div>
            <div class="kpi-sub" style="color: #0284c7 !important;">Total Weight: {total_weight} pts</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br/>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. FOUR CORE TABS
# -----------------------------------------------------------------------------
tab_map, tab_match, tab_arena, tab_dossier = st.tabs([
    "Geospatial Atlas & Leaderboard",
    "City Livability Matchmaker",
    "Head-to-Head Comparison",
    "Comprehensive City Dossier",
])

# =============================================================================
# TAB 1: GEOSPATIAL ATLAS & LEADERBOARD
# =============================================================================
with tab_map:
    col_map, col_table = st.columns([1.5, 1.0])

    with col_map:
        if is_national:
            fig_map = create_india_map(ranked_df, score_col="custom_score")
        else:
            fig_map = create_karnataka_map(ranked_df, score_col="custom_score")
        st.plotly_chart(fig_map, width="stretch")

    with col_table:
        st.markdown(f"#### Dynamic {tier_label} Leaderboard")

        # Search filter
        search_query = st.text_input("Search City / District:", placeholder="Type to filter e.g. Bengaluru, Mysuru...").strip().lower()

        filtered_table = display_df.copy()
        if search_query:
            filtered_table = filtered_table[filtered_table["name"].str.lower().str.contains(search_query)]

        cols_show = ["custom_rank", "name"]
        if not is_national:
            cols_show.append("region")
        else:
            cols_show.append("state")
        cols_show.extend(["custom_score", "rank_change"])

        view_table = filtered_table[cols_show].copy()
        view_table.rename(
            columns={
                "custom_rank": "Rank",
                "name": "City / District",
                "state": "State",
                "region": "Sub-Region",
                "custom_score": "Score",
                "rank_change": "Shift",
            },
            inplace=True,
        )

        st.dataframe(
            view_table,
            height=480,
            width="stretch",
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Livability Score", min_value=0, max_value=100, format="%.1f"
                ),
                "Shift": st.column_config.NumberColumn(
                    "Shift vs Base", format="%+d"
                ),
            },
        )

        csv_bytes = ranked_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Ranked Dataset (CSV)",
            data=csv_bytes,
            file_name=f"livability_index_{tier_label.lower()}.csv",
            mime="text/csv",
            width="stretch",
        )

# =============================================================================
# TAB 2: CITY LIVABILITY MATCHMAKER
# =============================================================================
with tab_match:
    st.markdown("### Algorithmic City Matchmaker")
    st.caption("Select your lifestyle preferences to compute personalized recommendation scores across candidate cities:")

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("#### 1. Career & Life Phase")
        career_focus = st.selectbox(
            "Primary focus:",
            options=["Tech & High-Growth Professional", "University / College Student", "Remote Worker / Digital Nomad", "Family with School-Going Children", "Retiree / Senior Citizen"],
        )
    with m_col2:
        st.markdown("#### 2. Cost & Budget Expectations")
        financial_focus = st.selectbox(
            "Budget priority:",
            options=["High Budget (Willing to pay for premier amenities)", "Moderate Budget (Seeking balanced cost-of-living)", "Strict Budget (Maximizing affordability & low rent)"],
        )
    with m_col3:
        st.markdown("#### 3. Environmental Priority")
        env_focus = st.selectbox(
            "Key requirement:",
            options=["Pristine Clean Air & Low Pollution", "Pleasant Mild Climate & Nature", "High Public Safety & Low Crime Rate", "Top Multi-Speciality Healthcare Access"],
        )

    # Derive automated preference weights from the quiz
    quiz_weights: Dict[str, float] = user_weights.copy()
    if "Student" in career_focus:
        quiz_weights["Education"] = 10.0
        quiz_weights["Affordability"] = 9.0
    elif "Remote" in career_focus:
        quiz_weights["Connectivity"] = 10.0
        quiz_weights["Affordability"] = 9.0
    elif "Retiree" in career_focus:
        quiz_weights["Healthcare"] = 10.0
        quiz_weights["Climate"] = 9.0
    elif "Family" in career_focus:
        quiz_weights["Safety"] = 10.0
        quiz_weights["Healthcare"] = 9.0
        quiz_weights["Air Quality"] = 9.0

    if "Strict" in financial_focus:
        quiz_weights["Affordability"] = 10.0
    elif "High Budget" in financial_focus:
        quiz_weights["Connectivity"] = max(quiz_weights["Connectivity"], 9.0)

    if "Pristine Clean Air" in env_focus:
        quiz_weights["Air Quality"] = 10.0
    elif "Pleasant Mild Climate" in env_focus:
        quiz_weights["Climate"] = 10.0
    elif "Public Safety" in env_focus:
        quiz_weights["Safety"] = 10.0
    elif "Healthcare" in env_focus:
        quiz_weights["Healthcare"] = 10.0

    matches = find_best_match_cities(active_dataset, quiz_weights, top_n=3)

    st.markdown("---")
    st.markdown("#### Top Recommended Matches")

    match_cols = st.columns(3)
    choice_labels = ["Rank 1 Recommendation", "Rank 2 Recommendation", "Rank 3 Recommendation"]
    card_borders = ["#0284c7", "#10b981", "#8b5cf6"]

    for idx, match in enumerate(matches):
        with match_cols[idx]:
            st.markdown(
                f"""
                <div class="match-card" style="border-top: 4px solid {card_borders[idx]};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 700; color: #475569 !important; font-size: 0.85rem;">{choice_labels[idx]}</span>
                        <span class="match-ring">{match['match_percent']}% Fit</span>
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a !important; margin-bottom: 2px;">{match['name']}</div>
                    <div class="match-meta">{match['region']} • {match['tier']}</div>
                    <div class="match-score">Composite Score: {match['custom_score']:.1f}/100</div>
                    <div style="margin-bottom: 8px;">
                        <span class="badge-superpower">Primary Strength:</span>
                        <span style="font-size: 0.82rem; font-weight: 600; color: #1e293b !important;">{match['superpower']}</span>
                    </div>
                    <div style="margin-bottom: 12px;">
                        <span class="badge-bottleneck">Key Consideration:</span>
                        <span style="font-size: 0.82rem; font-weight: 600; color: #1e293b !important;">{match['bottleneck']}</span>
                    </div>
                    <div class="match-archetype">
                        <b>Classification:</b> {match.get('tier', 'Urban Center')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =============================================================================
# TAB 3: HEAD-TO-HEAD ARENA
# =============================================================================
with tab_arena:
    st.markdown("### Head-to-Head Comparative Arena")
    st.caption("Compare two cities or districts across all 7 livability dimensions:")

    all_names = ranked_df["name"].tolist()
    arena_c1, arena_c2 = st.columns(2)
    with arena_c1:
        city_a = st.selectbox("Select Contender A (Primary):", options=all_names, index=0)
    with arena_c2:
        city_b = st.selectbox("Select Contender B (Comparison):", options=all_names, index=1 if len(all_names) > 1 else 0)

    id_a = ranked_df[ranked_df["name"] == city_a]["id"].iloc[0]
    id_b = ranked_df[ranked_df["name"] == city_b]["id"].iloc[0]

    meta, comp_df = compare_two_entities(ranked_df, id_a, id_b)
    row_a = ranked_df[ranked_df["id"] == id_a].iloc[0]
    row_b = ranked_df[ranked_df["id"] == id_b].iloc[0]

    # Calculate head-to-head win tally
    leads_a = sum(comp_df["Leader"] == city_a)
    leads_b = sum(comp_df["Leader"] == city_b)
    ties = sum(comp_df["Leader"] == "Tie")

    matchup_winner = city_a if meta["entity_a_score"] > meta["entity_b_score"] else (city_b if meta["entity_b_score"] > meta["entity_a_score"] else "Draw")

    st.markdown(
        f"""
        <div class="arena-card">
            <div style="flex: 1;">
                <div class="arena-name-a">{city_a}</div>
                <div class="arena-score-a">{meta['entity_a_score']:.1f}</div>
                <div class="arena-sub-a">Rank #{int(row_a['custom_rank'])} • Leads in {leads_a}/7 Pillars</div>
            </div>
            <div style="flex: 0.4;" class="arena-vs">VS</div>
            <div style="flex: 1;">
                <div class="arena-name-b">{city_b}</div>
                <div class="arena-score-b">{meta['entity_b_score']:.1f}</div>
                <div class="arena-sub-b">Rank #{int(row_b['custom_rank'])} • Leads in {leads_b}/7 Pillars</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    arena_left, arena_right = st.columns([1.1, 1.1])
    with arena_left:
        fig_dual_radar = create_comparison_radar_chart(row_a, row_b)
        st.plotly_chart(fig_dual_radar, width="stretch")

    with arena_right:
        fig_diff = create_diverging_diff_bar(row_a, row_b)
        st.plotly_chart(fig_diff, width="stretch")

    st.markdown("#### Detailed Dimension Comparison Matrix")
    display_comp = comp_df[["Pillar", f"{city_a} (Score)", f"{city_b} (Score)", "Diff (A - B)", "Leader"]]
    st.dataframe(display_comp, hide_index=True, width="stretch")

# =============================================================================
# TAB 4: COMPREHENSIVE CITY DOSSIER
# =============================================================================
with tab_dossier:
    st.markdown("### City / District Comprehensive Dossier")
    picked_city = st.selectbox("Select Entity to Inspect:", options=ranked_df["name"].tolist(), index=0)
    city_row = ranked_df[ranked_df["name"] == picked_city].iloc[0]

    # Generate narrative dossier
    narrative = generate_city_narrative(city_row, ranked_df)

    # Narrative callout card
    st.markdown(
        f"""
        <div class="dossier-card">
            <div class="dossier-title">Executive Dossier: {picked_city}</div>
            <div class="dossier-summary">{narrative['summary']}</div>
            <div style="display: flex; gap: 24px; font-size: 0.85rem; flex-wrap: wrap;">
                <div><span class="badge-superpower">Key Strengths:</span> <b style="color: #0f172a !important;">{narrative['key_strengths']}</b></div>
                <div><span class="badge-bottleneck">Key Considerations:</span> <b style="color: #0f172a !important;">{narrative['key_bottlenecks']}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dos_left, dos_right = st.columns([1.1, 1.1])
    with dos_left:
        # Benchmark average series
        benchmark_row = ranked_df[SCORE_COLUMNS].mean()
        fig_radar_bench = create_radar_chart(city_row, benchmark_row=benchmark_row)
        st.plotly_chart(fig_radar_bench, width="stretch")

    with dos_right:
        st.markdown(f"#### {picked_city} vs. {tier_label} Benchmark")
        detail_rows = []
        for p, col in PILLAR_TO_SCORE_COL.items():
            raw_c = PILLAR_TO_RAW_COL.get(p)
            score_val = float(city_row[col])
            benchmark_val = float(ranked_df[col].mean())
            raw_val = city_row[raw_c] if raw_c and raw_c in city_row else "N/A"

            detail_rows.append({
                "Pillar": p,
                "City Score": score_val,
                "Benchmark": np.round(benchmark_val, 1),
                "Delta vs Avg": np.round(score_val - benchmark_val, 1),
                "Raw Value": raw_val,
            })

        detail_df = pd.DataFrame(detail_rows)
        st.dataframe(
            detail_df,
            hide_index=True,
            width="stretch",
            column_config={
                "City Score": st.column_config.ProgressColumn("City Score", min_value=0, max_value=100, format="%.1f"),
                "Delta vs Avg": st.column_config.NumberColumn("Margin", format="%+.1f"),
            },
        )

# -----------------------------------------------------------------------------
# 7. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 12px 0;'>
        <b>India City Livability Index & Karnataka State Deep-Dive</b> | Master of Computer Applications (MCA) Project<br>
        BMS Institute of Technology & Management (BMSITM), Bengaluru • Academic Year 2026<br>
        Developed with <b>Python, Streamlit, Plotly & Scikit-Learn</b>
    </div>
    """,
    unsafe_allow_html=True,
)
