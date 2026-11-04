"""Unit tests for Module 3 Calculator Engine."""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calculator import (
    calculate_livability_index,
    compare_two_entities,
    get_top_ranked,
    find_best_match_cities,
    generate_city_narrative,
    PERSONA_PRESETS,
    PILLAR_TO_SCORE_COL,
)


def get_sample_df():
    data = {
        "id": ["C1", "C2", "C3"],
        "name": ["City Alpha", "City Beta", "City Gamma"],
        "region": ["North", "South", "East"],
        "tier": ["Tier 1", "Tier 2", "Tier 2"],
        "baseline_rank": [1, 2, 3],
        "baseline_score": [70.0, 60.0, 50.0],
        "score_climate": [80.0, 60.0, 40.0],
        "score_air_quality": [90.0, 50.0, 60.0],
        "score_safety": [70.0, 80.0, 50.0],
        "score_education": [85.0, 65.0, 45.0],
        "score_affordability": [40.0, 75.0, 90.0],
        "score_healthcare": [75.0, 60.0, 50.0],
        "score_connectivity": [95.0, 50.0, 30.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_df():
    return get_sample_df()


def test_balanced_persona_matches_mean(sample_df):
    """Equal weights across all 7 pillars should match the arithmetic mean."""
    res = calculate_livability_index(sample_df, weights="Balanced")
    score_cols = list(PILLAR_TO_SCORE_COL.values())
    expected_means = np.round(sample_df[score_cols].mean(axis=1), 2)

    # Sort sample_df identically to match res index
    for idx, row in res.iterrows():
        orig_row = sample_df[sample_df["id"] == row["id"]].iloc[0]
        assert np.isclose(row["custom_score"], round(float(orig_row[score_cols].mean()), 2))


def test_student_persona_favors_affordable_education(sample_df):
    """Student persona should reward high education and affordability."""
    res = calculate_livability_index(sample_df, weights="Student")
    assert "custom_score" in res.columns
    assert "custom_rank" in res.columns
    assert "rank_change" in res.columns
    assert res["custom_rank"].iloc[0] == 1


def test_custom_weights_dictionary(sample_df):
    """Passing a custom dictionary of weights works correctly."""
    custom_w = {
        "Affordability": 10.0,
        "Air Quality": 0.0,
        "Safety": 0.0,
        "Education": 0.0,
        "Climate": 0.0,
        "Healthcare": 0.0,
        "Connectivity": 0.0,
    }
    res = calculate_livability_index(sample_df, weights=custom_w)
    # When only Affordability has weight, City Gamma (affordability=90) should rank #1
    assert res.iloc[0]["id"] == "C3"
    assert res.iloc[0]["custom_score"] == 90.0
    assert res.iloc[0]["custom_rank"] == 1


def test_rank_change_calculation(sample_df):
    """Rank change should correctly report difference from baseline_rank."""
    # Weight only affordability: C3 was baseline 3, now rank 1 -> rank_change = +2
    custom_w = {"Affordability": 10.0}
    res = calculate_livability_index(sample_df, weights=custom_w)
    c3 = res[res["id"] == "C3"].iloc[0]
    assert c3["custom_rank"] == 1
    assert c3["rank_change"] == 2  # baseline 3 - custom 1 = +2


def test_error_all_zero_weights(sample_df):
    """Zero sum of weights must raise ValueError."""
    zero_w = {k: 0.0 for k in PILLAR_TO_SCORE_COL.keys()}
    with pytest.raises(ValueError, match="Sum of weights must be strictly greater than zero"):
        calculate_livability_index(sample_df, weights=zero_w)


def test_error_negative_weight(sample_df):
    """Negative weights must raise ValueError."""
    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_livability_index(sample_df, weights={"Climate": -5.0})


def test_error_unknown_persona(sample_df):
    """Unknown persona string must raise KeyError."""
    with pytest.raises(KeyError, match="Unknown persona"):
        calculate_livability_index(sample_df, weights="NonExistentPersona")


def test_error_unknown_dimension_name(sample_df):
    """Unknown dimension key in weights dict must raise KeyError."""
    with pytest.raises(KeyError, match="Unknown livability dimension"):
        calculate_livability_index(sample_df, weights={"FantasyScore": 10.0})


def test_error_non_numeric_weight(sample_df):
    """Non-numeric weight values must raise TypeError."""
    with pytest.raises(TypeError, match="must be numeric"):
        calculate_livability_index(sample_df, weights={"Climate": "five"})


def test_compare_two_entities(sample_df):
    """Head to head comparison returns metadata and comparison DataFrame."""
    meta, comp_df = compare_two_entities(sample_df, "C1", "C2")
    assert meta["entity_a_id"] == "C1"
    assert meta["entity_b_id"] == "C2"
    assert len(comp_df) == 7
    assert "Leader" in comp_df.columns
    assert "Pillar" in comp_df.columns


def test_compare_two_entities_invalid_id(sample_df):
    """Invalid entity IDs in comparison must raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        compare_two_entities(sample_df, "INVALID_ID", "C1")


def test_get_top_ranked_with_filters(sample_df):
    """Top ranked helper handles n and tier filters."""
    res = calculate_livability_index(sample_df, weights="Balanced")
    top_1 = get_top_ranked(res, n=1)
    assert len(top_1) == 1

    tier_2 = get_top_ranked(res, n=5, tier="Tier 2")
    assert all(tier_2["tier"] == "Tier 2")


def test_find_best_match_cities(sample_df):
    """Matchmaker returns top matches with match percent and superpower tags."""
    weights = {"Education": 10.0, "Affordability": 9.0}
    matches = find_best_match_cities(sample_df, weights, top_n=2)
    assert len(matches) == 2
    assert "match_percent" in matches[0]
    assert "superpower" in matches[0]
    assert "bottleneck" in matches[0]
    assert matches[0]["match_percent"] > 50.0


def test_generate_city_narrative(sample_df):
    """Narrative generator creates summary, strengths, and weaknesses."""
    row = sample_df.iloc[0]
    narrative = generate_city_narrative(row, sample_df)
    assert "summary" in narrative
    assert "key_strengths" in narrative
    assert "key_bottlenecks" in narrative
    assert row["name"] in narrative["summary"]


if __name__ == "__main__":
    pytest_funcs = [
        test_balanced_persona_matches_mean,
        test_student_persona_favors_affordable_education,
        test_custom_weights_dictionary,
        test_rank_change_calculation,
        test_error_all_zero_weights,
        test_error_negative_weight,
        test_error_unknown_persona,
        test_error_unknown_dimension_name,
        test_error_non_numeric_weight,
        test_compare_two_entities,
        test_compare_two_entities_invalid_id,
        test_get_top_ranked_with_filters,
    ]

    print("=" * 75)
    print("  RUNNING MODULE 3 TESTS")
    print("=" * 75)
    df = get_sample_df()
    passed = 0
    for f in pytest_funcs:
        try:
            f(df)
            print(f" [PASS] {f.__name__}")
            passed += 1
        except Exception as e:
            print(f" [FAIL] {f.__name__}: {e}")

    print("=" * 75)
    print(f"  MODULE 3 TEST SUMMARY: {passed} / {len(pytest_funcs)} PASSED.")
    print("=" * 75)
