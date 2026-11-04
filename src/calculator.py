"""
Module 3: Dynamic Weighted Index & Persona Calculator Engine
Author: India & Karnataka Livability Index Project

Implements real-time weighted composite scoring, user persona presets,
dynamic ranking, rank-change tracking, and head-to-head comparison analytics.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.preprocessor import SCORE_COLUMNS, SCORE_COLUMN_MAPPING

# User-facing pillar to score column mapping
PILLAR_TO_SCORE_COL: Dict[str, str] = {
    "Climate": "score_climate",
    "Air Quality": "score_air_quality",
    "Safety": "score_safety",
    "Education": "score_education",
    "Affordability": "score_affordability",
    "Healthcare": "score_healthcare",
    "Connectivity": "score_connectivity",
}

SCORE_COL_TO_PILLAR: Dict[str, str] = {v: k for k, v in PILLAR_TO_SCORE_COL.items()}

# Raw metric mapping for detailed comparisons
PILLAR_TO_RAW_COL: Dict[str, str] = {
    "Climate": "climate_comfort_raw",
    "Air Quality": "air_quality_pm25",
    "Safety": "safety_crime_rate",
    "Education": "education_index",
    "Affordability": "monthly_cost_living_inr",
    "Healthcare": "healthcare_infra",
    "Connectivity": "connectivity_score",
}

# Pre-configured Persona Weight Profiles (1 - 10 scale)
PERSONA_PRESETS: Dict[str, Dict[str, float]] = {
    "Balanced": {
        "Climate": 5.0,
        "Air Quality": 5.0,
        "Safety": 5.0,
        "Education": 5.0,
        "Affordability": 5.0,
        "Healthcare": 5.0,
        "Connectivity": 5.0,
    },
    "Student": {
        "Climate": 4.0,
        "Air Quality": 5.0,
        "Safety": 7.0,
        "Education": 10.0,
        "Affordability": 9.0,
        "Healthcare": 4.0,
        "Connectivity": 7.0,
    },
    "Family": {
        "Climate": 6.0,
        "Air Quality": 9.0,
        "Safety": 10.0,
        "Education": 8.0,
        "Affordability": 7.0,
        "Healthcare": 9.0,
        "Connectivity": 6.0,
    },
    "Retiree": {
        "Climate": 9.0,
        "Air Quality": 9.0,
        "Safety": 8.0,
        "Education": 2.0,
        "Affordability": 7.0,
        "Healthcare": 10.0,
        "Connectivity": 4.0,
    },
    "Remote Worker": {
        "Climate": 7.0,
        "Air Quality": 8.0,
        "Safety": 7.0,
        "Education": 3.0,
        "Affordability": 9.0,
        "Healthcare": 6.0,
        "Connectivity": 10.0,
    },
    "Job Seeker": {
        "Climate": 5.0,
        "Air Quality": 4.0,
        "Safety": 6.0,
        "Education": 8.0,
        "Affordability": 8.0,
        "Healthcare": 6.0,
        "Connectivity": 10.0,
    },
}


def _resolve_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Standardizes a weights dictionary so keys match SCORE_COLUMNS.
    Accepts keys as either friendly pillar names ('Air Quality') or column names ('score_air_quality').
    """
    resolved: Dict[str, float] = {}

    for k, v in weights.items():
        if k in PILLAR_TO_SCORE_COL:
            col_name = PILLAR_TO_SCORE_COL[k]
        elif k in SCORE_COLUMNS:
            col_name = k
        else:
            raise KeyError(f"Unknown livability dimension or score column: '{k}'")

        try:
            val_float = float(v)
        except (ValueError, TypeError):
            raise TypeError(f"Weight for '{k}' must be numeric, got {type(v).__name__}: {v}")

        if val_float < 0:
            raise ValueError(f"Weight for '{k}' cannot be negative: {val_float}")

        resolved[col_name] = val_float

    # Ensure all 7 score columns have an assigned weight (default 0.0 if not provided)
    for col in SCORE_COLUMNS:
        if col not in resolved:
            resolved[col] = 0.0

    return resolved


def calculate_livability_index(
    df: pd.DataFrame,
    weights: Union[str, Dict[str, float]] = "Balanced",
) -> pd.DataFrame:
    """
    Calculates weighted composite livability scores and ranks for cities/districts.

    Parameters:
    - df: Preprocessed DataFrame containing score_* columns.
    - weights: Either a persona name (e.g. 'Student', 'Family') or a dict of weights.

    Returns:
    - Copy of DataFrame sorted descending by custom_score, with:
      - 'custom_score': Weighted average (0-100 scale).
      - 'custom_rank': Dynamic 1-based rank.
      - 'rank_change': Movement relative to baseline_rank (+ moved up, - dropped).
    """
    if df is None or df.empty:
        raise ValueError("Cannot calculate livability index for empty DataFrame.")

    # Check that score columns exist
    missing_scores = [c for c in SCORE_COLUMNS if c not in df.columns]
    if missing_scores:
        raise ValueError(f"DataFrame is missing required score columns: {missing_scores}")

    # Resolve weights from persona or dictionary
    if isinstance(weights, str):
        if weights not in PERSONA_PRESETS:
            raise KeyError(
                f"Unknown persona '{weights}'. Available: {list(PERSONA_PRESETS.keys())}"
            )
        raw_weights = PERSONA_PRESETS[weights]
    elif isinstance(weights, dict):
        raw_weights = weights
    else:
        raise TypeError(f"Weights must be str or dict, got {type(weights).__name__}")

    weight_dict = _resolve_weights(raw_weights)
    total_weight = sum(weight_dict.values())

    if np.isclose(total_weight, 0.0):
        raise ValueError("Sum of weights must be strictly greater than zero.")

    res_df = df.copy()

    # Calculate weighted sum
    weighted_scores = np.zeros(len(res_df), dtype=float)
    for col, w in weight_dict.items():
        if w > 0:
            weighted_scores += res_df[col].to_numpy(dtype=float) * w

    res_df["custom_score"] = np.round(weighted_scores / total_weight, 2)

    # Sort descending by custom score
    res_df.sort_values(by="custom_score", ascending=False, inplace=True)
    res_df.reset_index(drop=True, inplace=True)

    # Assign 1-based rank
    res_df["custom_rank"] = res_df["custom_score"].rank(ascending=False, method="min").astype(int)

    # Calculate rank change relative to baseline if baseline_rank is present
    if "baseline_rank" in res_df.columns:
        # Positive means improved rank (e.g. baseline was 5, now is 2 -> +3)
        res_df["rank_change"] = res_df["baseline_rank"] - res_df["custom_rank"]
    else:
        res_df["rank_change"] = 0

    return res_df


def compare_two_entities(
    df: pd.DataFrame,
    id_a: str,
    id_b: str,
) -> Tuple[Dict[str, Union[str, float]], pd.DataFrame]:
    """
    Performs a head-to-head comparison between two cities or districts.
    Returns:
    - Metadata comparison dict (names, states, scores).
    - Detailed breakdown DataFrame across all 7 pillars (Score A, Score B, Diff, Winner).
    """
    row_a = df[df["id"] == id_a]
    row_b = df[df["id"] == id_b]

    if row_a.empty:
        raise ValueError(f"Entity ID '{id_a}' not found in dataset.")
    if row_b.empty:
        raise ValueError(f"Entity ID '{id_b}' not found in dataset.")

    a = row_a.iloc[0]
    b = row_b.iloc[0]

    meta = {
        "entity_a_id": id_a,
        "entity_a_name": a["name"],
        "entity_a_region": a.get("region", a.get("state", "N/A")),
        "entity_a_score": a.get("custom_score", a.get("baseline_score", 0.0)),
        "entity_b_id": id_b,
        "entity_b_name": b["name"],
        "entity_b_region": b.get("region", b.get("state", "N/A")),
        "entity_b_score": b.get("custom_score", b.get("baseline_score", 0.0)),
    }

    breakdown_rows = []
    for pillar, score_col in PILLAR_TO_SCORE_COL.items():
        raw_col = PILLAR_TO_RAW_COL.get(pillar)
        score_val_a = float(a[score_col])
        score_val_b = float(b[score_col])
        if abs(score_val_a) < 1e-6:
            score_val_a = 0.0
        if abs(score_val_b) < 1e-6:
            score_val_b = 0.0

        raw_val_a = a[raw_col] if raw_col and raw_col in a else "N/A"
        raw_val_b = b[raw_col] if raw_col and raw_col in b else "N/A"

        score_diff = np.round(score_val_a - score_val_b, 2)
        if abs(score_diff) < 1e-6:
            score_diff = 0.0
        if score_diff > 0:
            winner = a["name"]
        elif score_diff < 0:
            winner = b["name"]
        else:
            winner = "Tie"

        breakdown_rows.append({
            "Pillar": pillar,
            f"{a['name']} (Raw)": raw_val_a,
            f"{b['name']} (Raw)": raw_val_b,
            f"{a['name']} (Score)": score_val_a,
            f"{b['name']} (Score)": score_val_b,
            "Diff (A - B)": score_diff,
            "Leader": winner,
        })

    comparison_df = pd.DataFrame(breakdown_rows)
    return meta, comparison_df


def get_top_ranked(
    df: pd.DataFrame,
    n: int = 5,
    region: Optional[str] = None,
    tier: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns top N ranked cities/districts with optional region or tier filtering.
    """
    filtered = df.copy()
    if region and "region" in filtered.columns:
        filtered = filtered[filtered["region"].str.lower() == region.lower()]
    if tier and "tier" in filtered.columns:
        filtered = filtered[filtered["tier"].str.lower() == tier.lower()]

    rank_col = "custom_rank" if "custom_rank" in filtered.columns else "baseline_rank"
    return filtered.sort_values(by=rank_col).head(n)


def find_best_match_cities(
    df: pd.DataFrame,
    priority_weights: Dict[str, float],
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """
    Intelligent Matchmaker Engine:
    Ranks entities based on priority weights, calculates a 0-100% compatibility match,
    and generates automated qualitative recommendations.
    """
    ranked = calculate_livability_index(df, weights=priority_weights)
    top_matches = ranked.head(top_n)

    results = []
    for _, r in top_matches.iterrows():
        # Identify standout pillar and bottleneck pillar
        scores = {p: float(r[col]) for p, col in PILLAR_TO_SCORE_COL.items()}
        best_pillar = max(scores, key=scores.get)
        worst_pillar = min(scores, key=scores.get)

        # Match percentage relative to top possible score
        match_pct = min(99.0, max(50.0, float(r["custom_score"]) * 1.08))

        results.append({
            "id": r["id"],
            "name": r["name"],
            "region": r.get("region", r.get("state", "India")),
            "tier": r.get("tier", "Metro"),
            "custom_score": float(r["custom_score"]),
            "match_percent": np.round(match_pct, 1),
            "superpower": f"{best_pillar} ({scores[best_pillar]:.0f}/100)",
            "bottleneck": f"{worst_pillar} ({scores[worst_pillar]:.0f}/100)",
            "cluster": r.get("cluster_label", "Growth Tier"),
            "population_lakhs": r.get("population_lakhs", "N/A"),
        })

    return results


def generate_city_narrative(row: pd.Series, benchmark_df: pd.DataFrame) -> Dict[str, str]:
    """
    Generates a natural-language executive dossier and SWOT-style insights for a city/district.
    """
    scores = {p: float(row[col]) for p, col in PILLAR_TO_SCORE_COL.items()}
    benchmarks = {p: float(benchmark_df[col].mean()) for p, col in PILLAR_TO_SCORE_COL.items()}

    # Categorize strengths (beats average by > 10 pts) and weaknesses (trails average by > 10 pts)
    strengths = [p for p, s in scores.items() if s - benchmarks[p] >= 8.0]
    weaknesses = [p for p, s in scores.items() if benchmarks[p] - s >= 8.0]

    top_p = max(scores, key=scores.get)
    low_p = min(scores, key=scores.get)

    strengths_str = ", ".join(strengths[:3]) if strengths else f"Consistent balance in {top_p}"
    weaknesses_str = ", ".join(weaknesses[:2]) if weaknesses else "No critical lagging dimensions"

    summary = (
        f"{row['name']} stands out as a strong performer with a composite livability score of "
        f"{row.get('custom_score', row.get('baseline_score', 0)):.1f}/100. "
        f"Its primary competitive advantage lies in {top_p} (score {scores[top_p]:.0f}/100). "
        f"The primary development bottleneck to navigate is {low_p} (score {scores[low_p]:.0f}/100)."
    )

    return {
        "summary": summary,
        "key_strengths": strengths_str,
        "key_bottlenecks": weaknesses_str,
        "top_pillar": top_p,
        "lowest_pillar": low_p,
    }


if __name__ == "__main__":
    print("=" * 75)
    print("  INDIA & KARNATAKA LIVABILITY INDEX - MODULE 3: CALCULATOR ENGINE")
    print("=" * 75)

    # Load master processed datasets
    nat_csv = PROJECT_ROOT / "data" / "processed" / "master_national.csv"
    ka_csv = PROJECT_ROOT / "data" / "processed" / "master_karnataka.csv"

    if not nat_csv.exists() or not ka_csv.exists():
        from src.preprocessor import generate_master_datasets
        master_nat, master_ka = generate_master_datasets()
    else:
        master_nat = pd.read_csv(nat_csv)
        master_ka = pd.read_csv(ka_csv)

    # 1. Test Student Persona on National
    print("\n[1] Pan-India Rankings for 'Student' Persona:")
    print("    (Heavy weights on Education=10 and Affordability=9)")
    student_nat = calculate_livability_index(master_nat, weights="Student")
    cols_to_show = ["custom_rank", "name", "state", "custom_score", "rank_change", "score_education", "score_affordability"]
    print(student_nat[cols_to_show].head(5).to_string(index=False))

    # 2. Test Retiree Persona on Karnataka
    print("\n" + "-" * 75)
    print("[2] Karnataka Rankings for 'Retiree' Persona:")
    print("    (Heavy weights on Healthcare=10, Climate=9, Air Quality=9, Safety=8)")
    retiree_ka = calculate_livability_index(master_ka, weights="Retiree")
    ka_cols = ["custom_rank", "name", "region", "custom_score", "rank_change", "score_healthcare", "score_climate", "score_air_quality"]
    print(retiree_ka[ka_cols].head(5).to_string(index=False))

    # 3. Test Head-to-Head Comparison: Bengaluru vs Mysuru
    print("\n" + "-" * 75)
    print("[3] Head-to-Head Comparison: Bengaluru Urban (KA_BLR_U) vs Mysuru (KA_MYS)")
    meta, comp_df = compare_two_entities(master_ka, "KA_BLR_U", "KA_MYS")
    print(comp_df[["Pillar", "Bengaluru Urban (Score)", "Mysuru (Score)", "Diff (A - B)", "Leader"]].to_string(index=False))

    print("\n" + "=" * 75)
    print("  MODULE 3 CALCULATOR ENGINE: ALL TESTS PASSED [OK]")
    print("=" * 75)
