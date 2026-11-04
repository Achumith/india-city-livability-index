"""
Comprehensive Test Suite for Modules 1 & 2
Checklist:
1. Functional correctness.
2. Expected and unexpected inputs.
3. Error handling.
4. Preserving existing functionality.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_national_data,
    load_karnataka_data,
    validate_schema,
    get_data_summary,
    REQUIRED_IDENTIFIER_COLS,
    RAW_METRIC_COLS,
    METRIC_DIRECTION,
)
from src.preprocessor import (
    normalize_metric,
    impute_missing_values,
    process_dataset,
    generate_master_datasets,
    SCORE_COLUMNS,
)


# =========================================================================
# 1. FUNCTIONAL CORRECTNESS TESTS
# =========================================================================

def test_1_1_national_and_karnataka_loading_integrity():
    """Verify correct row count, non-null identifiers, and uniqueness."""
    df_nat = load_national_data()
    df_ka = load_karnataka_data()

    assert len(df_nat) == 30, f"Expected 30 cities, got {len(df_nat)}"
    assert len(df_ka) == 31, f"Expected 31 districts, got {len(df_ka)}"

    assert df_nat["id"].is_unique, "National IDs must be unique"
    assert df_ka["id"].is_unique, "Karnataka IDs must be unique"
    assert df_nat["name"].is_unique, "National city names must be unique"
    assert df_ka["name"].is_unique, "Karnataka district names must be unique"


def test_1_2_normalization_scaling_correctness():
    """Verify MinMax formulas for both higher_is_better and lower_is_better."""
    # Test higher is better
    s_high = pd.Series([10.0, 30.0, 50.0])
    norm_high = normalize_metric(s_high, direction="higher_is_better")
    assert norm_high.tolist() == [0.0, 50.0, 100.0]

    # Test lower is better (inversion)
    s_low = pd.Series([10.0, 30.0, 50.0])
    norm_low = normalize_metric(s_low, direction="lower_is_better")
    assert norm_low.tolist() == [100.0, 50.0, 0.0]


def test_1_3_baseline_score_calculation():
    """Verify baseline unweighted score is strictly the arithmetic mean of the 7 dimension scores."""
    df_nat = load_national_data()
    processed = process_dataset(df_nat, "National")

    expected_mean = np.round(processed[SCORE_COLUMNS].mean(axis=1), 2)
    diff = np.abs(processed["baseline_score"] - expected_mean)
    assert (diff < 1e-4).all(), "Baseline score must match mean of score columns"


def test_1_4_ranking_order_correctness():
    """Verify ranking is in strictly descending order of baseline score."""
    df_ka = load_karnataka_data()
    processed = process_dataset(df_ka, "Karnataka")

    scores = processed["baseline_score"].tolist()
    assert scores == sorted(scores, reverse=True), "Dataset must be sorted descending by baseline_score"
    assert processed["baseline_rank"].iloc[0] == 1, "Top rank must be 1"


# =========================================================================
# 2. EXPECTED AND UNEXPECTED INPUTS TESTS
# =========================================================================

def test_2_1_single_row_input():
    """Single row inputs have min == max, must scale to neutral 50.0 without crashing."""
    s_single = pd.Series([42.0])
    res_high = normalize_metric(s_single, direction="higher_is_better")
    res_low = normalize_metric(s_single, direction="lower_is_better")
    assert res_high.iloc[0] == 50.0
    assert res_low.iloc[0] == 50.0


def test_2_2_constant_column_input():
    """Columns where all cities have the exact same value should yield 50.0."""
    s_const = pd.Series([100.0, 100.0, 100.0, 100.0])
    res = normalize_metric(s_const, direction="higher_is_better")
    assert (res == 50.0).all()


def test_2_3_all_nan_series_input():
    """All-NaN input series should fall back to neutral 50.0 rather than raising exceptions."""
    s_nan = pd.Series([np.nan, np.nan, np.nan])
    res = normalize_metric(s_nan, direction="higher_is_better")
    assert (res == 50.0).all()


def test_2_4_empty_series_input():
    """Empty series should return an empty series without crashing."""
    s_empty = pd.Series([], dtype=float)
    res = normalize_metric(s_empty, direction="higher_is_better")
    assert res.empty


def test_2_5_all_nan_column_imputation():
    """If an entire column is NaN, impute_missing_values falls back to 50.0 neutral midpoint."""
    df = pd.DataFrame({"air_quality_pm25": [np.nan, np.nan, np.nan]})
    df_imp, report = impute_missing_values(df, ["air_quality_pm25"])
    assert (df_imp["air_quality_pm25"] == 50.0).all()
    assert report["air_quality_pm25"] == 3


def test_2_6_non_numeric_metric_detection():
    """Schema validation must reject non-numeric values (e.g. strings or invalid tokens)."""
    df_bad = load_national_data().copy()
    df_bad["air_quality_pm25"] = df_bad["air_quality_pm25"].astype(str)
    df_bad.loc[0, "air_quality_pm25"] = "BAD_STRING"
    is_valid, errors = validate_schema(df_bad, "BadNational")
    assert not is_valid
    assert any("must be numeric" in err for err in errors)


def test_2_7_out_of_bounds_coordinates():
    """Schema validation must reject coordinates outside India's geographic bounds."""
    df_bad = load_national_data().copy()
    df_bad.loc[0, "latitude"] = 75.0  # Latitude 75° is North of the Arctic Circle!
    is_valid, errors = validate_schema(df_bad, "BadCoordinates")
    assert not is_valid
    assert any("latitude outside India" in err for err in errors)


# =========================================================================
# 3. ERROR HANDLING TESTS
# =========================================================================

def test_3_1_load_non_existent_file():
    """Calling load_national_data with a non-existent path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_national_data(Path("does_not_exist_file.csv"))


def test_3_2_invalid_normalization_direction():
    """Passing an unknown direction to normalize_metric must raise ValueError."""
    s = pd.Series([10.0, 20.0, 30.0])
    with pytest.raises(ValueError, match="Invalid direction"):
        normalize_metric(s, direction="random_direction")


def test_3_3_process_empty_dataframe():
    """Passing an empty DataFrame to process_dataset must raise ValueError."""
    with pytest.raises(ValueError, match="Cannot process empty"):
        process_dataset(pd.DataFrame(), "EmptyTest")


def test_3_4_process_missing_metric_columns():
    """Passing a DataFrame missing required metric columns to process_dataset must raise ValueError."""
    df_incomplete = pd.DataFrame({"name": ["CityA", "CityB"]})
    with pytest.raises(ValueError, match="Missing required metric columns"):
        process_dataset(df_incomplete, "IncompleteTest")


def test_3_5_impute_missing_non_existent_column():
    """Passing non-existent column names to impute_missing_values must raise KeyError."""
    df = pd.DataFrame({"val": [1, 2, 3]})
    with pytest.raises(KeyError, match="not found"):
        impute_missing_values(df, ["non_existent_column"])


# =========================================================================
# 4. EXISTING FUNCTIONALITY INTEGRATION TEST
# =========================================================================

def test_4_1_generate_master_datasets_preserves_end_to_end():
    """Ensures generate_master_datasets creates compliant CSVs without regression."""
    master_nat, master_ka = generate_master_datasets()

    assert len(master_nat) == 30
    assert len(master_ka) == 31

    for df, name in [(master_nat, "National"), (master_ka, "Karnataka")]:
        for col in SCORE_COLUMNS:
            assert col in df.columns
            assert (df[col] >= 0.0).all() and (df[col] <= 100.0).all(), f"Scores out of bounds in {name} {col}"
        assert "baseline_score" in df.columns
        assert "baseline_rank" in df.columns
        assert df["baseline_rank"].iloc[0] == 1

    # Check files exist on disk
    nat_csv = PROJECT_ROOT / "data" / "processed" / "master_national.csv"
    ka_csv = PROJECT_ROOT / "data" / "processed" / "master_karnataka.csv"
    assert nat_csv.exists(), "master_national.csv must exist"
    assert ka_csv.exists(), "master_karnataka.csv must exist"


if __name__ == "__main__":
    tests = [
        test_1_1_national_and_karnataka_loading_integrity,
        test_1_2_normalization_scaling_correctness,
        test_1_3_baseline_score_calculation,
        test_1_4_ranking_order_correctness,
        test_2_1_single_row_input,
        test_2_2_constant_column_input,
        test_2_3_all_nan_series_input,
        test_2_4_empty_series_input,
        test_2_5_all_nan_column_imputation,
        test_2_6_non_numeric_metric_detection,
        test_2_7_out_of_bounds_coordinates,
        test_3_1_load_non_existent_file,
        test_3_2_invalid_normalization_direction,
        test_3_3_process_empty_dataframe,
        test_3_4_process_missing_metric_columns,
        test_3_5_impute_missing_non_existent_column,
        test_4_1_generate_master_datasets_preserves_end_to_end,
    ]

    print("=" * 75)
    print("  RUNNING COMPREHENSIVE TESTS FOR MODULES 1 & 2")
    print("=" * 75)
    passed = 0
    for t in tests:
        try:
            t()
            print(f" [PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f" [FAIL] {t.__name__}: {e}")

    print("=" * 75)
    print(f"  TEST SUMMARY: {passed} / {len(tests)} TESTS PASSED.")
    print("=" * 75)
    if passed == len(tests):
        print("  ALL RIGOROUS CHECKS SATISFIED [OK]")
    else:
        sys.exit(1)
