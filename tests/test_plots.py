"""Unit tests for Module 5 Visualization Engine (Plotly)."""

import sys
from pathlib import Path
import pytest
import pandas as pd
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plots import (
    create_india_map,
    create_karnataka_map,
    create_radar_chart,
    create_comparison_radar_chart,
    create_diverging_diff_bar,
    create_pca_cluster_scatter,
    create_leaderboard_bar,
)
from src.ml_engine import run_full_ml_pipeline


@pytest.fixture
def loaded_data():
    nat_csv = PROJECT_ROOT / "data" / "processed" / "master_national.csv"
    ka_csv = PROJECT_ROOT / "data" / "processed" / "master_karnataka.csv"
    df_nat = pd.read_csv(nat_csv)
    df_ka = pd.read_csv(ka_csv)
    return df_nat, df_ka


def test_create_india_map(loaded_data):
    df_nat, _ = loaded_data
    fig = create_india_map(df_nat)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert fig.data[0].type == "scattergeo"


def test_create_karnataka_map(loaded_data):
    _, df_ka = loaded_data
    fig = create_karnataka_map(df_ka)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    # Trace types could be choropleth and/or scattergeo
    trace_types = [t.type for t in fig.data]
    assert "scattergeo" in trace_types or "choropleth" in trace_types


def test_create_radar_charts(loaded_data):
    df_nat, _ = loaded_data
    row_a = df_nat.iloc[0]
    row_b = df_nat.iloc[1]

    fig_single = create_radar_chart(row_a)
    assert isinstance(fig_single, go.Figure)
    assert fig_single.data[0].type == "scatterpolar"

    fig_comp = create_comparison_radar_chart(row_a, row_b)
    assert isinstance(fig_comp, go.Figure)
    assert len(fig_comp.data) == 2
    assert fig_comp.data[0].type == "scatterpolar"
    assert fig_comp.data[1].type == "scatterpolar"

    fig_diff = create_diverging_diff_bar(row_a, row_b)
    assert isinstance(fig_diff, go.Figure)
    assert len(fig_diff.data) == 1
    assert fig_diff.data[0].type == "bar"


def test_create_pca_cluster_scatter(loaded_data):
    df_nat, _ = loaded_data
    enriched_nat, _ = run_full_ml_pipeline(df_nat, "National", n_clusters=3)
    fig = create_pca_cluster_scatter(enriched_nat)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert fig.data[0].type == "scatter"


def test_create_leaderboard_bar(loaded_data):
    df_nat, _ = loaded_data
    fig = create_leaderboard_bar(df_nat, top_n=5)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert fig.data[0].type == "bar"


if __name__ == "__main__":
    nat_csv = PROJECT_ROOT / "data" / "processed" / "master_national.csv"
    ka_csv = PROJECT_ROOT / "data" / "processed" / "master_karnataka.csv"
    data = (pd.read_csv(nat_csv), pd.read_csv(ka_csv))

    test_create_india_map(data)
    test_create_karnataka_map(data)
    test_create_radar_charts(data)
    test_create_pca_cluster_scatter(data)
    test_create_leaderboard_bar(data)
    print("All Module 5 Plotly Tests Passed Successfully!")
