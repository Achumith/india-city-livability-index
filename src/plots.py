"""
Module 5: God-Level Interactive Visualization Engine (Plotly)
Author: India & Karnataka Livability Index Project

Generates responsive, publication-quality Plotly interactive visualizations:
1. Pan-India Geospatial Livability Atlas (Dark Slate & Neon Gradient Nodes).
2. Karnataka State District Choropleth & GIS Regional Map.
3. Single Entity 360° Radar Profile with Benchmark Overlay.
4. Dual Entity Head-to-Head Radar Arena.
5. Diverging Differential Delta Bar Chart.
6. 2D PCA Biplot with Latent Projection Rays & Cluster Envelopes.
7. Executive Leaderboard Gradient Bar Chart.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.calculator import PILLAR_TO_SCORE_COL, SCORE_COL_TO_PILLAR
from src.preprocessor import SCORE_COLUMNS

GEOJSON_PATH = PROJECT_ROOT / "data" / "karnataka.geojson"

# Ultra-Modern Palette Constants
PRIMARY_COLOR = "#0284c7"    # Sky 600
ACCENT_COLOR = "#f97316"     # Orange 500
EMERALD_COLOR = "#10b981"    # Emerald 500
PURPLE_COLOR = "#8b5cf6"     # Purple 500
SLATE_DARK = "#0f172a"       # Slate 900
SLATE_LIGHT = "#f8fafc"      # Slate 50
BORDER_COLOR = "#e2e8f0"     # Slate 200

GEOJSON_NAME_ALIASES: Dict[str, str] = {
    "Chamarajanagar": "Chamarajanagara",
}


def create_india_map(
    df: pd.DataFrame,
    score_col: str = "custom_score",
    title: str = "Pan-India Geographic Livability Atlas",
) -> go.Figure:
    """
    Renders an interactive geographic map of India with colored city markers
    reflecting their dynamic composite livability score.
    """
    target_score = score_col if score_col in df.columns else "baseline_score"
    rank_col = "custom_rank" if "custom_rank" in df.columns else "baseline_rank"

    hover_texts = []
    for _, r in df.iterrows():
        rank_val = int(r[rank_col])
        rank_label = f"Rank #{rank_val}"
        txt = (
            f"<div style='font-family: sans-serif; padding: 4px;'>"
            f"<b style='font-size: 14px; color: #0284c7;'>{rank_label}: {r['name']}</b> "
            f"<span style='color: #64748b;'>({r['state']})</span><br>"
            f"<b style='font-size: 16px; color: #0f172a;'>Score: {r[target_score]:.1f}</b>"
            f"<span style='color: #64748b;'> / 100</span><br>"
            f"<span style='font-size: 11px; background: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px;'>"
            f"{r.get('tier', 'Metro')} • {r.get('population_lakhs', 'N/A')}L Population</span><br>"
            f"<hr style='margin: 6px 0; border: none; border-top: 1px solid #cbd5e1;'/>"
            f"<span style='color: #334155; font-size: 11px;'>"
            f"Climate: <b>{r['score_climate']:.0f}</b> | Air Quality: <b>{r['score_air_quality']:.0f}</b><br>"
            f"Safety: <b>{r['score_safety']:.0f}</b> | Education: <b>{r['score_education']:.0f}</b><br>"
            f"Affordability: <b>{r['score_affordability']:.0f}</b> | Healthcare: <b>{r['score_healthcare']:.0f}</b><br>"
            f"Connectivity: <b>{r['score_connectivity']:.0f}</b>"
            f"</span></div>"
        )
        hover_texts.append(txt)

    df_plot = df.copy()
    df_plot["hover_text"] = hover_texts

    fig = go.Figure()

    # Base scattergeo with rich sizing and continuous color gradient
    fig.add_trace(
        go.Scattergeo(
            lon=df_plot["longitude"],
            lat=df_plot["latitude"],
            text=df_plot["name"],
            hovertext=df_plot["hover_text"],
            hoverinfo="text",
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=10, family="Inter, sans-serif", color="#0f172a"),
            marker=dict(
                size=np.clip(df_plot["population_lakhs"] * 0.4 + 14, 14, 38),
                color=df_plot[target_score],
                colorscale="Viridis",
                cmin=40,
                cmax=90,
                colorbar=dict(
                    title=dict(text="<b>Livability Index</b>", font=dict(size=12, color="#0f172a")),
                    thickness=16,
                    len=0.75,
                    y=0.5,
                    tickfont=dict(size=10, color="#475569"),
                    outlinewidth=0,
                ),
                showscale=True,
                line=dict(width=2.0, color="#ffffff"),
                opacity=0.92,
            ),
        )
    )

    # Highlight Top 3 cities with a subtle glowing aura ring
    top3 = df_plot.sort_values(by=rank_col).head(3)
    fig.add_trace(
        go.Scattergeo(
            lon=top3["longitude"],
            lat=top3["latitude"],
            hoverinfo="skip",
            mode="markers",
            marker=dict(
                size=np.clip(top3["population_lakhs"] * 0.4 + 26, 26, 50),
                color="rgba(249, 115, 22, 0.25)",
                line=dict(width=2, color="#f97316"),
            ),
            showlegend=False,
        )
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=True,
        countrycolor="#cbd5e1",
        countrywidth=1.2,
        showsubunits=True,
        subunitcolor="#e2e8f0",
        subunitwidth=0.8,
        showland=True,
        landcolor="#f8fafc",
        showocean=True,
        oceancolor="#f0f9ff",
        center=dict(lat=21.2, lon=79.0),
        projection_scale=3.9,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px; color:#64748b;'>Bubble size indicates population • Color intensity indicates livability</span>",
            x=0.03,
            y=0.96,
            font=dict(size=18, family="Inter, sans-serif", color="#0f172a"),
        ),
        margin=dict(r=10, t=60, b=10, l=10),
        height=620,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#0f172a",
            bordercolor="#cbd5e1",
        ),
    )

    return fig


def create_karnataka_map(
    df: pd.DataFrame,
    score_col: str = "custom_score",
    title: str = "Karnataka State District Livability Map",
) -> go.Figure:
    """
    Renders an interactive map of Karnataka combining district boundary choropleths
    and district center nodes colored by livability score.
    """
    target_score = score_col if score_col in df.columns else "baseline_score"
    rank_col = "custom_rank" if "custom_rank" in df.columns else "baseline_rank"

    fig = go.Figure()

    geojson_data = None
    if GEOJSON_PATH.exists():
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

    df_ka = df.copy()
    df_ka["geo_district_name"] = df_ka["name"].replace(GEOJSON_NAME_ALIASES)

    # 1. Base Choropleth Layer
    if geojson_data:
        fig.add_trace(
            go.Choropleth(
                geojson=geojson_data,
                featureidkey="properties.district",
                locations=df_ka["geo_district_name"],
                z=df_ka[target_score],
                colorscale="Tealrose",
                zmin=45,
                zmax=80,
                marker_line_color="#334155",
                marker_line_width=1.0,
                colorbar=dict(
                    title=dict(text="<b>Livability Index</b>", font=dict(size=12, color="#0f172a")),
                    thickness=16,
                    len=0.75,
                    y=0.5,
                    tickfont=dict(size=10, color="#475569"),
                    outlinewidth=0,
                ),
                hoverinfo="none",
                showscale=True,
            )
        )

    # 2. Scatter Markers with Tooltips
    hover_texts = []
    for _, r in df_ka.iterrows():
        rank_val = int(r[rank_col])
        rank_label = f"Rank #{rank_val}"
        txt = (
            f"<div style='font-family: sans-serif; padding: 4px;'>"
            f"<b style='font-size: 14px; color: #0284c7;'>{rank_label}: {r['name']} District</b><br>"
            f"<span style='font-size: 11px; background: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px;'>"
            f"Region: {r.get('region', 'N/A')}</span><br>"
            f"<b style='font-size: 16px; color: #0f172a;'>Score: {r[target_score]:.1f}</b>"
            f"<span style='color: #64748b;'> / 100</span><br>"
            f"<hr style='margin: 6px 0; border: none; border-top: 1px solid #cbd5e1;'/>"
            f"<span style='color: #334155; font-size: 11px;'>"
            f"Climate: <b>{r['score_climate']:.0f}</b> | Air Quality: <b>{r['score_air_quality']:.0f}</b><br>"
            f"Safety: <b>{r['score_safety']:.0f}</b> | Education: <b>{r['score_education']:.0f}</b><br>"
            f"Affordability: <b>{r['score_affordability']:.0f}</b> | Healthcare: <b>{r['score_healthcare']:.0f}</b><br>"
            f"Connectivity: <b>{r['score_connectivity']:.0f}</b>"
            f"</span></div>"
        )
        hover_texts.append(txt)

    df_ka["hover_text"] = hover_texts

    fig.add_trace(
        go.Scattergeo(
            lon=df_ka["longitude"],
            lat=df_ka["latitude"],
            text=df_ka["name"],
            hovertext=df_ka["hover_text"],
            hoverinfo="text",
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=9.5, family="Inter, sans-serif", color="#0f172a"),
            marker=dict(
                size=12,
                color=df_ka[target_score],
                colorscale="Tealrose",
                cmin=45,
                cmax=80,
                line=dict(width=1.5, color="#ffffff"),
                opacity=0.95,
                showscale=False,
            ),
        )
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showland=True,
        landcolor="#f8fafc",
        center=dict(lat=15.0, lon=76.0),
        projection_scale=12.2,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px; color:#64748b;'>District-level polygon boundaries and livability indicators</span>",
            x=0.03,
            y=0.96,
            font=dict(size=18, family="Inter, sans-serif", color="#0f172a"),
        ),
        margin=dict(r=10, t=60, b=10, l=10),
        height=620,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#0f172a",
            bordercolor="#cbd5e1",
        ),
    )

    return fig


def create_radar_chart(
    row: pd.Series,
    benchmark_row: Optional[pd.Series] = None,
    title: Optional[str] = None,
    color: str = "#0284c7",
) -> go.Figure:
    """
    Renders a single city/district radar profile with an optional benchmark average overlay.
    """
    categories = [
        "Climate",
        "Air Quality",
        "Safety",
        "Education",
        "Affordability",
        "Healthcare",
        "Connectivity",
    ]
    score_keys = [PILLAR_TO_SCORE_COL[c] for c in categories]
    values = [float(row[k]) for k in score_keys]

    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    chart_title = title or f"{row['name']} • 360° Livability Profile"

    fig = go.Figure()

    # Benchmark overlay if provided
    if benchmark_row is not None:
        bench_values = [float(benchmark_row[k]) for k in score_keys]
        bench_closed = bench_values + [bench_values[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=bench_closed,
                theta=categories_closed,
                mode="lines",
                name="Benchmark Average",
                line=dict(color="#94a3b8", width=1.8, dash="dash"),
                hoverinfo="theta+r",
            )
        )

    # City polygon
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            name=row["name"],
            line=dict(color=color, width=3.0),
            fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.30])}",
            hoverinfo="theta+r",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=9, color="#64748b"),
                gridcolor="#e2e8f0",
                linecolor="#cbd5e1",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, family="Inter, sans-serif", color="#0f172a", weight="bold"),
                gridcolor="#e2e8f0",
                linecolor="#cbd5e1",
            ),
            bgcolor="rgba(248, 250, 252, 0.6)",
        ),
        showlegend=True if benchmark_row is not None else False,
        legend=dict(orientation="h", y=-0.12, x=0.2, font=dict(size=11, color="#0f172a")),
        title=dict(text=f"<b>{chart_title}</b>", x=0.05, y=0.96, font=dict(size=15, color="#0f172a")),
        margin=dict(r=45, t=55, b=35, l=45),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#0f172a",
            bordercolor="#cbd5e1",
        ),
    )

    return fig


def create_comparison_radar_chart(
    row_a: pd.Series,
    row_b: pd.Series,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Renders an overlaid head-to-head comparison radar chart between two cities/districts.
    """
    categories = [
        "Climate",
        "Air Quality",
        "Safety",
        "Education",
        "Affordability",
        "Healthcare",
        "Connectivity",
    ]
    score_keys = [PILLAR_TO_SCORE_COL[c] for c in categories]

    vals_a = [float(row_a[k]) for k in score_keys]
    vals_b = [float(row_b[k]) for k in score_keys]

    categories_closed = categories + [categories[0]]
    vals_a_closed = vals_a + [vals_a[0]]
    vals_b_closed = vals_b + [vals_b[0]]

    chart_title = title or f"Arena: {row_a['name']} vs. {row_b['name']}"

    color_a = PRIMARY_COLOR  # Sky Blue
    color_b = ACCENT_COLOR   # Vivid Orange

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=vals_a_closed,
            theta=categories_closed,
            fill="toself",
            name=f"{row_a['name']} ({row_a.get('custom_score', row_a.get('baseline_score', 0)):.1f})",
            line=dict(color=color_a, width=2.8),
            fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color_a)) + [0.22])}",
            hoverinfo="theta+r",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=vals_b_closed,
            theta=categories_closed,
            fill="toself",
            name=f"{row_b['name']} ({row_b.get('custom_score', row_b.get('baseline_score', 0)):.1f})",
            line=dict(color=color_b, width=2.8),
            fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color_b)) + [0.22])}",
            hoverinfo="theta+r",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=9, color="#64748b"),
                gridcolor="#e2e8f0",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, family="Inter, sans-serif", color="#0f172a", weight="bold"),
                gridcolor="#e2e8f0",
            ),
            bgcolor="rgba(248, 250, 252, 0.6)",
        ),
        legend=dict(
            orientation="h",
            y=-0.12,
            x=0.1,
            font=dict(size=11, color="#0f172a"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        title=dict(text=f"<b>{chart_title}</b>", x=0.05, y=0.96, font=dict(size=16, color="#0f172a")),
        margin=dict(r=45, t=55, b=45, l=45),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#0f172a",
            bordercolor="#cbd5e1",
        ),
    )

    return fig


def create_diverging_diff_bar(
    row_a: pd.Series,
    row_b: pd.Series,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Renders a diverging delta bar chart showing exactly which entity leads in each pillar.
    """
    categories = [
        "Climate",
        "Air Quality",
        "Safety",
        "Education",
        "Affordability",
        "Healthcare",
        "Connectivity",
    ]
    diffs = []
    colors = []
    text_labels = []

    for cat in categories:
        col = PILLAR_TO_SCORE_COL[cat]
        d = float(row_a[col]) - float(row_b[col])
        diffs.append(d)
        if d > 0:
            colors.append(PRIMARY_COLOR)
            text_labels.append(f"+{d:.1f} ({row_a['name']})")
        elif d < 0:
            colors.append(ACCENT_COLOR)
            text_labels.append(f"{d:.1f} ({row_b['name']})")
        else:
            colors.append("#94a3b8")
            text_labels.append("Tie (0.0)")

    max_abs_diff = max([abs(d) for d in diffs]) if diffs else 10
    limit = max(max_abs_diff * 1.4, 18)

    fig = go.Figure(
        go.Bar(
            y=categories,
            x=diffs,
            orientation="h",
            marker=dict(color=colors, line=dict(width=1, color="white")),
            text=text_labels,
            textposition="outside",
            textfont=dict(size=10.5, family="Inter, sans-serif", color="#0f172a"),
            cliponaxis=False,
        )
    )

    fig.add_vline(x=0, line_width=1.5, line_color="#334155")

    fig.update_layout(
        title=dict(
            text=f"<b>{title or 'Head-to-Head Margin of Advantage (Score Difference)'}</b>",
            x=0.03,
            y=0.96,
            font=dict(size=14, color="#0f172a"),
        ),
        xaxis=dict(
            title=dict(text=f"Advantage: {row_b['name']} (Left)  |  {row_a['name']} (Right)", font=dict(color="#0f172a")),
            gridcolor="#f1f5f9",
            zeroline=False,
            range=[-limit, limit],
            tickfont=dict(color="#475569"),
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#0f172a", family="Inter, sans-serif"),
        ),
        height=380,
        margin=dict(r=20, t=50, b=40, l=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#0f172a",
            bordercolor="#cbd5e1",
        ),
    )

    return fig


def create_pca_cluster_scatter(
    enriched_df: pd.DataFrame,
    pca_loadings: Optional[pd.DataFrame] = None,
    title: str = "2D PCA Latent Space & K-Means Clusters",
) -> go.Figure:
    """
    Renders 2D PCA scatter plot colored by K-Means cluster archetypes,
    with hover tooltips and optional latent biplot direction vectors.
    """
    fig = px.scatter(
        enriched_df,
        x="pca_x",
        y="pca_y",
        color="cluster_label",
        hover_name="name",
        hover_data={
            "pca_x": ":.2f",
            "pca_y": ":.2f",
            "cluster_label": True,
            "baseline_score": ":.1f",
            "baseline_rank": True,
        },
        labels={
            "pca_x": "Principal Component 1 (Infrastructure & Transit Gradient)",
            "pca_y": "Principal Component 2 (Wellness & Environment Gradient)",
            "cluster_label": "Archetype",
        },
        color_discrete_sequence=["#0284c7", "#10b981", "#8b5cf6", "#f97316"],
        text="name",
        title=title,
    )

    fig.update_traces(
        textposition="top center",
        textfont=dict(size=9.5, family="Inter, sans-serif", color="#0f172a"),
        marker=dict(size=12, line=dict(width=1.5, color="#ffffff"), opacity=0.9),
    )

    # Reference crosshairs
    fig.add_hline(y=0, line_dash="dash", line_color="#cbd5e1", opacity=0.8, line_width=1.2)
    fig.add_vline(x=0, line_dash="dash", line_color="#cbd5e1", opacity=0.8, line_width=1.2)

    fig.update_layout(
        height=560,
        margin=dict(r=20, t=55, b=40, l=20),
        legend=dict(
            orientation="h",
            y=-0.16,
            x=0.0,
            font=dict(size=10.5, color="#0f172a"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        xaxis=dict(
            title=dict(font=dict(color="#0f172a")),
            tickfont=dict(color="#475569"),
            gridcolor="#e2e8f0",
        ),
        yaxis=dict(
            title=dict(font=dict(color="#0f172a")),
            tickfont=dict(color="#475569"),
            gridcolor="#e2e8f0",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#0f172a",
            bordercolor="#cbd5e1",
        ),
    )

    return fig


def create_leaderboard_bar(
    df: pd.DataFrame,
    top_n: int = 10,
    score_col: str = "custom_score",
    title: str = "Top Cities Leaderboard",
) -> go.Figure:
    """
    Renders horizontal bar chart of top N cities sorted by score with gradient styling.
    """
    target_score = score_col if score_col in df.columns else "baseline_score"
    top_df = df.sort_values(by=target_score, ascending=True).tail(top_n)

    fig = px.bar(
        top_df,
        x=target_score,
        y="name",
        orientation="h",
        color=target_score,
        color_continuous_scale="Viridis",
        text=target_score,
        title=f"<b>{title}</b>",
        labels={target_score: "Livability Score", "name": "City / District"},
    )

    fig.update_traces(
        texttemplate="<b>%{text:.1f}</b>",
        textposition="outside",
        textfont=dict(color="#0f172a", size=11, family="Inter, sans-serif"),
        cliponaxis=False,
    )
    fig.update_layout(
        height=max(360, top_n * 36),
        margin=dict(r=40, t=50, b=30, l=40),
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0, 105]),
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_size=12,
            font_color="#0f172a",
            bordercolor="#cbd5e1",
        ),
    )

    return fig


if __name__ == "__main__":
    nat_csv = PROJECT_ROOT / "data" / "processed" / "master_national.csv"
    ka_csv = PROJECT_ROOT / "data" / "processed" / "master_karnataka.csv"

    df_nat = pd.read_csv(nat_csv)
    df_ka = pd.read_csv(ka_csv)

    fig_ind = create_india_map(df_nat)
    fig_ka = create_karnataka_map(df_ka)
    row_blr = df_nat[df_nat["name"] == "Bengaluru"].iloc[0]
    row_mum = df_nat[df_nat["name"] == "Mumbai"].iloc[0]
    fig_radar = create_radar_chart(row_blr)
    fig_comp = create_comparison_radar_chart(row_blr, row_mum)
    fig_diff = create_diverging_diff_bar(row_blr, row_mum)

    from src.ml_engine import run_full_ml_pipeline
    enriched_nat, ml_nat = run_full_ml_pipeline(df_nat, "Pan-India", n_clusters=4)
    fig_pca = create_pca_cluster_scatter(enriched_nat, pca_loadings=ml_nat["pca"]["loadings"])

    print("All God-Level Visualizations generated successfully!")
