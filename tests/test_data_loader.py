"""Unit tests for Module 1 Data Loader."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.data_loader import (
    load_national_data,
    load_karnataka_data,
    REQUIRED_IDENTIFIER_COLS,
    RAW_METRIC_COLS,
    validate_schema,
)


def test_national_data_loading():
    df = load_national_data()
    assert len(df) == 30, f"Expected 30 national cities, got {len(df)}"
    assert all(col in df.columns for col in REQUIRED_IDENTIFIER_COLS)
    assert all(col in df.columns for col in RAW_METRIC_COLS)
    assert df["id"].is_unique, "National city IDs must be unique"
    assert df["name"].is_unique, "National city names must be unique"


def test_karnataka_data_loading():
    df = load_karnataka_data()
    assert len(df) == 31, f"Expected 31 Karnataka districts, got {len(df)}"
    assert all(col in df.columns for col in REQUIRED_IDENTIFIER_COLS)
    assert all(col in df.columns for col in RAW_METRIC_COLS)
    assert df["id"].is_unique, "Karnataka district IDs must be unique"
    assert df["name"].is_unique, "Karnataka district names must be unique"


def test_schema_validation_rejects_empty():
    import pandas as pd
    empty_df = pd.DataFrame()
    is_valid, errors = validate_schema(empty_df, "Empty")
    assert not is_valid
    assert len(errors) > 0


if __name__ == "__main__":
    test_national_data_loading()
    test_karnataka_data_loading()
    test_schema_validation_rejects_empty()
    print("All Module 1 Unit Tests Passed Successfully!")
