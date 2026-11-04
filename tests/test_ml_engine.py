"""Unit tests for Module 4 ML Analytics Engine (PCA & K-Means)."""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_engine import (
    run_pca_analysis,
    find_optimal_k,
    run_clustering_analysis,
    run_full_ml_pipeline,
)
from src.preprocessor import SCORE_COLUMNS


def get_mock_df(n: int = 15) -> pd.DataFrame:
    np.random.seed(42)
    data = {
        "id": [f"ID_{i}" for i in range(n)],
        "name": [f"City_{i}" for i in range(n)],
    }
    for col in SCORE_COLUMNS:
        data[col] = np.random.uniform(20.0, 95.0, size=n).round(2)
    return pd.DataFrame(data)


def test_pca_analysis_dimensions_and_variance():
    df = get_mock_df(20)
    pca_res = run_pca_analysis(df, n_components=2)

    assert "coords_2d" in pca_res
    assert pca_res["coords_2d"].shape == (20, 2)
    assert "explained_variance_ratio" in pca_res
    assert len(pca_res["explained_variance_ratio"]) == 2
    assert pca_res["total_explained_variance"] > 0.0
    assert pca_res["total_explained_variance"] <= 100.0

    loadings = pca_res["loadings"]
    assert loadings.shape == (len(SCORE_COLUMNS), 2)


def test_find_optimal_k():
    df = get_mock_df(20)
    k_eval = find_optimal_k(df, k_range=range(2, 5))

    assert "best_k" in k_eval
    assert k_eval["best_k"] in [2, 3, 4]
    assert "evaluation_table" in k_eval
    assert len(k_eval["evaluation_table"]) == 3
    assert "silhouette_score" in k_eval["evaluation_table"].columns


def test_kmeans_clustering_assignments():
    df = get_mock_df(20)
    kmeans_res = run_clustering_analysis(df, n_clusters=3)

    assert len(kmeans_res["labels"]) == 20
    assert set(kmeans_res["labels"]).issubset({0, 1, 2})
    assert -1.0 <= kmeans_res["silhouette_score"] <= 1.0
    assert len(kmeans_res["archetypes"]) == 3
    assert all(isinstance(v, str) and len(v) > 0 for v in kmeans_res["archetypes"].values())


def test_full_ml_pipeline_enrichment():
    df = get_mock_df(20)
    enriched_df, ml_diag = run_full_ml_pipeline(df, "MockTest", n_clusters=3)

    assert "pca_x" in enriched_df.columns
    assert "pca_y" in enriched_df.columns
    assert "cluster_id" in enriched_df.columns
    assert "cluster_label" in enriched_df.columns
    assert enriched_df["cluster_id"].nunique() == 3
    assert enriched_df["cluster_label"].isnull().sum() == 0


def test_error_empty_dataframe():
    with pytest.raises(ValueError, match="Cannot run ML pipeline on empty"):
        run_full_ml_pipeline(pd.DataFrame(), "EmptyTest")


def test_error_missing_feature_columns():
    df_missing = pd.DataFrame({"name": ["City1", "City2"]})
    with pytest.raises(ValueError, match="Missing required feature columns"):
        run_pca_analysis(df_missing)


if __name__ == "__main__":
    tests = [
        test_pca_analysis_dimensions_and_variance,
        test_find_optimal_k,
        test_kmeans_clustering_assignments,
        test_full_ml_pipeline_enrichment,
        test_error_empty_dataframe,
        test_error_missing_feature_columns,
    ]

    print("=" * 75)
    print("  RUNNING MODULE 4 TESTS")
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
    print(f"  MODULE 4 TEST SUMMARY: {passed} / {len(tests)} PASSED.")
    print("=" * 75)
