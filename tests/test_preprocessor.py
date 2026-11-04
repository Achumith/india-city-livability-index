"""Unit tests for Module 2 Preprocessor and Normalization Engine."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.preprocessor import (
    normalize_metric,
    impute_missing_values,
    process_dataset,
    generate_master_datasets,
    SCORE_COLUMNS,
)


def test_normalize_metric_higher_is_better():
    s = pd.Series([10.0, 20.0, 30.0])
    norm = normalize_metric(s, direction="higher_is_better")
    assert norm.iloc[0] == 0.0
    assert norm.iloc[1] == 50.0
    assert norm.iloc[2] == 100.0


def test_normalize_metric_lower_is_better():
    # Lower raw value should receive higher score (e.g. crime or pollution)
    s = pd.Series([10.0, 20.0, 30.0])
    norm = normalize_metric(s, direction="lower_is_better")
    assert norm.iloc[0] == 100.0  # lowest gets 100
    assert norm.iloc[1] == 50.0
    assert norm.iloc[2] == 0.0    # highest gets 0


def test_imputation_handles_nans():
    df = pd.DataFrame({"val": [10.0, np.nan, 30.0]})
    df_imp, report = impute_missing_values(df, ["val"])
    assert df_imp["val"].isnull().sum() == 0
    assert df_imp["val"].iloc[1] == 20.0  # median of [10, 30] is 20
    assert report["val"] == 1


def test_master_datasets_generation():
    master_nat, master_ka = generate_master_datasets()

    for name, df in [("National", master_nat), ("Karnataka", master_ka)]:
        assert len(df) > 0, f"{name} dataset is empty"
        for col in SCORE_COLUMNS:
            assert col in df.columns, f"Missing score column: {col} in {name}"
            # All normalized scores must be within [0, 100]
            assert df[col].min() >= 0.0, f"Score under 0 in {name} {col}"
            assert df[col].max() <= 100.0, f"Score over 100 in {name} {col}"
            assert df[col].isnull().sum() == 0, f"Null score in {name} {col}"

        assert "baseline_score" in df.columns
        assert "baseline_rank" in df.columns
        assert df["baseline_rank"].iloc[0] == 1


if __name__ == "__main__":
    test_normalize_metric_higher_is_better()
    test_normalize_metric_lower_is_better()
    test_imputation_handles_nans()
    test_master_datasets_generation()
    print("All Module 2 Unit Tests Passed Successfully!")
