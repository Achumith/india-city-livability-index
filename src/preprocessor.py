"""
Module 2: Data Preprocessing, Normalization & Master Dataset Generator
Author: India & Karnataka Livability Index Project

Transforms raw multi-unit metrics into unified 0-100 scores using MinMaxScaler,
handles directional inversion for inverse indicators (PM2.5, crime, cost),
performs median imputation, and generates master processed datasets.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data_loader import (
    load_national_data,
    load_karnataka_data,
    LIVABILITY_PILLARS,
    RAW_METRIC_COLS,
    METRIC_DIRECTION,
    REQUIRED_IDENTIFIER_COLS,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Mapping from raw metric column to normalized score column (0-100)
SCORE_COLUMN_MAPPING: Dict[str, str] = {
    "climate_comfort_raw": "score_climate",
    "air_quality_pm25": "score_air_quality",       # Higher score = cleaner air
    "safety_crime_rate": "score_safety",           # Higher score = safer
    "education_index": "score_education",           # Higher score = better education
    "monthly_cost_living_inr": "score_affordability", # Higher score = more affordable
    "healthcare_infra": "score_healthcare",         # Higher score = better healthcare
    "connectivity_score": "score_connectivity",     # Higher score = better transit
}

SCORE_COLUMNS: List[str] = list(SCORE_COLUMN_MAPPING.values())


def impute_missing_values(df: pd.DataFrame, metric_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Imputes missing values in metric columns using column-wise median.
    If a column is entirely NaN, falls back to 50.0 (neutral midpoint).
    Returns the updated DataFrame and a dict reporting imputed count per column.
    """
    df_imputed = df.copy()
    imputed_counts: Dict[str, int] = {}

    for col in metric_cols:
        if col not in df_imputed.columns:
            raise KeyError(f"Metric column '{col}' not found in DataFrame.")

        null_count = int(df_imputed[col].isnull().sum())
        if null_count > 0:
            median_val = df_imputed[col].median()
            # If all values are NaN, median is NaN, fallback to neutral 50.0
            if pd.isna(median_val):
                median_val = 50.0
            df_imputed[col] = df_imputed[col].fillna(median_val)
            imputed_counts[col] = null_count
        else:
            imputed_counts[col] = 0

    return df_imputed, imputed_counts


def normalize_metric(series: pd.Series, direction: str = "higher_is_better") -> pd.Series:
    """
    Normalizes a numerical series to a 0-100 scale using MinMaxScaler.
    
    - If direction == 'higher_is_better': Score = norm * 100 (highest raw = 100)
    - If direction == 'lower_is_better':  Score = (1 - norm) * 100 (lowest raw = 100)
    """
    if direction not in ("higher_is_better", "lower_is_better"):
        raise ValueError(
            f"Invalid direction '{direction}'. Must be 'higher_is_better' or 'lower_is_better'."
        )

    if series is None or series.empty:
        return pd.Series(dtype=float, index=series.index if series is not None else None)

    # Handle all-NaN series
    non_null = series.dropna()
    if non_null.empty:
        return pd.Series(50.0, index=series.index)

    vals = series.to_numpy().reshape(-1, 1)
    min_val, max_val = float(np.nanmin(vals)), float(np.nanmax(vals))

    if np.isclose(max_val, min_val):
        # Degenerate case: all values in the column are identical
        return pd.Series(50.0, index=series.index)

    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled = scaler.fit_transform(vals).flatten()

    if direction == "lower_is_better":
        # Invert: best performer (lowest original) gets 100, worst gets 0
        scaled = 100.0 - scaled

    return pd.Series(np.round(scaled, 2), index=series.index)


def process_dataset(raw_df: pd.DataFrame, dataset_name: str = "Dataset") -> pd.DataFrame:
    """
    Runs full preprocessing pipeline on a raw livability dataset:
    1. Validates presence of all required raw metric columns.
    2. Median imputation for missing values.
    3. MinMaxScaler normalization to 0-100 for all 7 dimensions.
    4. Handles directionality (inverting lower-is-better metrics).
    5. Computes baseline unweighted composite score (mean of 7 scores).
    """
    if raw_df is None or raw_df.empty:
        raise ValueError(f"Cannot process empty DataFrame for {dataset_name}.")

    missing_cols = [c for c in RAW_METRIC_COLS if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required metric columns for {dataset_name}: {missing_cols}")

    df, imputation_report = impute_missing_values(raw_df, RAW_METRIC_COLS)

    # Calculate normalized 0-100 scores for each dimension
    for raw_col, score_col in SCORE_COLUMN_MAPPING.items():
        direction = METRIC_DIRECTION[raw_col]
        df[score_col] = normalize_metric(df[raw_col], direction=direction)

    # Add baseline unweighted average livability score
    df["baseline_score"] = np.round(df[SCORE_COLUMNS].mean(axis=1), 2)

    # Sort descending by baseline score
    df.sort_values(by="baseline_score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["baseline_rank"] = df["baseline_score"].rank(ascending=False, method="min").astype(int)

    return df


def generate_master_datasets(output_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates and saves both National and Karnataka processed master datasets.
    """
    out_dir = output_dir or DATA_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process National Dataset
    raw_nat = load_national_data()
    master_nat = process_dataset(raw_nat, dataset_name="Pan-India")
    nat_path = out_dir / "master_national.csv"
    master_nat.to_csv(nat_path, index=False)

    # 2. Process Karnataka Dataset
    raw_ka = load_karnataka_data()
    master_ka = process_dataset(raw_ka, dataset_name="Karnataka")
    ka_path = out_dir / "master_karnataka.csv"
    master_ka.to_csv(ka_path, index=False)

    return master_nat, master_ka


if __name__ == "__main__":
    print("=" * 75)
    print("  INDIA & KARNATAKA LIVABILITY INDEX - MODULE 2: NORMALIZATION ENGINE")
    print("=" * 75)

    master_nat, master_ka = generate_master_datasets()

    print("\n[1] Processed Pan-India Master Dataset:")
    print(f" -> Saved to: data/processed/master_national.csv ({len(master_nat)} cities)")
    print(" -> Top 5 Cities by Baseline Unweighted Livability Score:")
    print(master_nat[["baseline_rank", "name", "state", "baseline_score"] + SCORE_COLUMNS[:3]].head(5).to_string(index=False))

    print("\n" + "-" * 75)
    print("[2] Processed Karnataka Master Dataset:")
    print(f" -> Saved to: data/processed/master_karnataka.csv ({len(master_ka)} districts)")
    print(" -> Top 5 Districts by Baseline Unweighted Livability Score:")
    print(master_ka[["baseline_rank", "name", "region", "baseline_score"] + SCORE_COLUMNS[:3]].head(5).to_string(index=False))

    print("\n" + "=" * 75)
    print("  MODULE 2 PREPROCESSING & NORMALIZATION: COMPLETED SUCCESSFULLY [OK]")
    print("=" * 75)
