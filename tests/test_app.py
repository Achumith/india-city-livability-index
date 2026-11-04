"""Unit test for Streamlit app data bundle and loading integrity."""

import sys
from pathlib import Path
import pytest
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import load_all_data
from src.calculator import calculate_livability_index, PERSONA_PRESETS


def test_app_data_bundle():
    bundle = load_all_data()

    assert "raw_nat" in bundle
    assert "raw_ka" in bundle
    assert "enriched_nat" in bundle
    assert "enriched_ka" in bundle
    assert "ml_nat" in bundle
    assert "ml_ka" in bundle

    assert len(bundle["raw_nat"]) == 30
    assert len(bundle["raw_ka"]) == 31

    # Check enriched ML columns are present
    for key in ["enriched_nat", "enriched_ka"]:
        df = bundle[key]
        assert "pca_x" in df.columns
        assert "pca_y" in df.columns
        assert "cluster_id" in df.columns
        assert "cluster_label" in df.columns


def test_app_persona_execution():
    bundle = load_all_data()
    for persona_name in PERSONA_PRESETS.keys():
        ranked_nat = calculate_livability_index(bundle["enriched_nat"], weights=persona_name)
        assert len(ranked_nat) == 30
        assert "custom_rank" in ranked_nat.columns
        assert ranked_nat["custom_rank"].iloc[0] == 1

        ranked_ka = calculate_livability_index(bundle["enriched_ka"], weights=persona_name)
        assert len(ranked_ka) == 31
        assert "custom_rank" in ranked_ka.columns
        assert ranked_ka["custom_rank"].iloc[0] == 1


if __name__ == "__main__":
    test_app_data_bundle()
    test_app_persona_execution()
    print("All App Integration Tests Passed Successfully!")
