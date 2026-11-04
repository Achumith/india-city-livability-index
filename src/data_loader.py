"""
Module 1: Data Architecture & Ingestion
Author: India & Karnataka Livability Index Project

This module defines the unified data schema for livability dimensions,
loads raw national and state-level datasets, and validates schema integrity.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Unified schema definitions
REQUIRED_IDENTIFIER_COLS: List[str] = [
    "id",
    "name",
    "state",
    "region",
    "tier",
    "latitude",
    "longitude",
    "population_lakhs",
]

# The 7 Core Livability Pillars and their raw metric column names
LIVABILITY_PILLARS: Dict[str, str] = {
    "Climate": "climate_comfort_raw",
    "Air Quality": "air_quality_pm25",
    "Safety": "safety_crime_rate",
    "Education": "education_index",
    "Affordability": "monthly_cost_living_inr",
    "Healthcare": "healthcare_infra",
    "Connectivity": "connectivity_score",
}

RAW_METRIC_COLS: List[str] = list(LIVABILITY_PILLARS.values())

# Metric directionality
METRIC_DIRECTION: Dict[str, str] = {
    "climate_comfort_raw": "higher_is_better",
    "air_quality_pm25": "lower_is_better",       # Inverted in normalization
    "safety_crime_rate": "lower_is_better",      # Inverted in normalization
    "education_index": "higher_is_better",
    "monthly_cost_living_inr": "lower_is_better", # Inverted in normalization
    "healthcare_infra": "higher_is_better",
    "connectivity_score": "higher_is_better",
}


def validate_schema(df: pd.DataFrame, dataset_name: str = "Dataset") -> Tuple[bool, List[str]]:
    """
    Validates that a DataFrame adheres to the required schema:
    - Checks for non-empty DataFrame.
    - Checks presence of required identifier columns.
    - Checks presence of all 7 raw metric columns.
    - Validates numeric datatypes for coordinates and metrics.
    - Validates geographic coordinate bounds for India.
    - Checks for negative values in numeric metrics.
    """
    errors: List[str] = []

    if df is None or df.empty:
        errors.append(f"{dataset_name} DataFrame is empty or None.")
        return False, errors

    # Check identifiers
    for col in REQUIRED_IDENTIFIER_COLS:
        if col not in df.columns:
            errors.append(f"Missing identifier column: '{col}'")

    # Check metrics
    for col in RAW_METRIC_COLS:
        if col not in df.columns:
            errors.append(f"Missing raw metric column: '{col}'")

    if errors:
        return False, errors

    # Check nulls in critical identifier columns
    null_counts = df[REQUIRED_IDENTIFIER_COLS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        errors.append(f"Found null values in critical identifier columns: {cols_with_nulls.to_dict()}")

    # Validate coordinate boundaries (India roughly: Lat 6° to 38° N, Lon 68° to 98° E)
    if "latitude" in df.columns and "longitude" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["latitude"]):
            errors.append("Column 'latitude' must be numeric.")
        elif not pd.api.types.is_numeric_dtype(df["longitude"]):
            errors.append("Column 'longitude' must be numeric.")
        else:
            invalid_lat = df[(df["latitude"] < 6.0) | (df["latitude"] > 38.0)]
            if not invalid_lat.empty:
                errors.append(f"Found {len(invalid_lat)} rows with latitude outside India boundaries.")

            invalid_lon = df[(df["longitude"] < 68.0) | (df["longitude"] > 98.0)]
            if not invalid_lon.empty:
                errors.append(f"Found {len(invalid_lon)} rows with longitude outside India boundaries.")

    # Validate metric columns are strictly numeric and non-negative
    for col in RAW_METRIC_COLS:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Metric column '{col}' must be numeric, got dtype '{df[col].dtype}'.")
            else:
                negative_vals = df[df[col] < 0]
                if not negative_vals.empty:
                    errors.append(f"Metric column '{col}' contains negative values.")

    is_valid = len(errors) == 0
    return is_valid, errors


def load_national_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Loads and validates the Pan-India major cities raw dataset."""
    path = filepath or (DATA_RAW_DIR / "raw_national_cities.csv")
    if not path.exists():
        raise FileNotFoundError(f"National dataset not found at {path}")

    df = pd.read_csv(path)
    is_valid, errors = validate_schema(df, dataset_name="Pan-India")
    if not is_valid:
        raise ValueError(f"National dataset schema validation failed: {errors}")

    return df


def load_karnataka_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Loads and validates the Karnataka state districts raw dataset."""
    path = filepath or (DATA_RAW_DIR / "raw_karnataka_districts.csv")
    if not path.exists():
        raise FileNotFoundError(f"Karnataka dataset not found at {path}")

    df = pd.read_csv(path)
    is_valid, errors = validate_schema(df, dataset_name="Karnataka")
    if not is_valid:
        raise ValueError(f"Karnataka dataset schema validation failed: {errors}")

    return df


def get_data_summary(df: pd.DataFrame, title: str = "Dataset Summary") -> pd.DataFrame:
    """Generates a formatted summary table of raw metrics."""
    summary = df[RAW_METRIC_COLS].describe().T[["mean", "std", "min", "50%", "max"]]
    summary.rename(columns={"50%": "median"}, inplace=True)
    summary["direction"] = [METRIC_DIRECTION[col] for col in summary.index]
    return summary


if __name__ == "__main__":
    print("=" * 70)
    print("  INDIA & KARNATAKA LIVABILITY INDEX - MODULE 1: DATA INGESTION")
    print("=" * 70)

    # 1. Test National Dataset
    print("\n[1] Ingesting Pan-India Major Metros Dataset...")
    df_nat = load_national_data()
    print(f" -> Successfully loaded {len(df_nat)} cities across {df_nat['state'].nunique()} States/UTs.")
    print(f" -> Columns: {list(df_nat.columns)}")
    print("\nPan-India Metrics Summary:")
    print(get_data_summary(df_nat, "National"))

    # 2. Test Karnataka Dataset
    print("\n" + "-" * 70)
    print("[2] Ingesting Karnataka State Districts Dataset...")
    df_ka = load_karnataka_data()
    print(f" -> Successfully loaded {len(df_ka)} districts across {df_ka['region'].nunique()} sub-regions.")
    print(f" -> Regions covered: {sorted(df_ka['region'].unique().tolist())}")
    print("\nKarnataka Districts Metrics Summary:")
    print(get_data_summary(df_ka, "Karnataka"))

    print("\n" + "=" * 70)
    print("  MODULE 1 DATA ARCHITECTURE & INGESTION: ALL CHECKS PASSED [OK]")
    print("=" * 70)
