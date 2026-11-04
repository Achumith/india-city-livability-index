"""
Module 4: Machine Learning Analytics Engine (PCA & K-Means Clustering)
Author: India & Karnataka Livability Index Project

Performs unsupervised machine learning on multi-dimensional livability data:
1. Principal Component Analysis (PCA): Dimensionality reduction to 2D, feature loadings,
   and latent variance extraction.
2. K-Means Clustering: Natural city grouping, Silhouette score evaluation, centroid profiling,
   and automated cluster archetype labeling.
Supports both Pan-India Metros and Karnataka State Districts.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.calculator import PILLAR_TO_SCORE_COL, SCORE_COL_TO_PILLAR
from src.preprocessor import SCORE_COLUMNS


def run_pca_analysis(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_components: int = 2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes PCA on the 7 livability dimensions:
    - Standardizes features using StandardScaler.
    - Projects data into 2 principal components for 2D visualization.
    - Computes explained variance ratios and cumulative variance.
    - Extracts feature loadings (eigenvector weights) showing what each axis represents.
    """
    cols = feature_cols or SCORE_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for PCA: {missing}")

    X = df[cols].to_numpy(dtype=float)

    # Standardize features (mean=0, variance=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    coords_2d = pca.fit_transform(X_scaled)

    explained_var = np.round(pca.explained_variance_ratio_ * 100, 2)
    cumulative_var = np.round(np.sum(explained_var), 2)

    # Compute Feature Loadings DataFrame (eigenvectors * sqrt(eigenvalues))
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    friendly_feature_names = [SCORE_COL_TO_PILLAR.get(c, c) for c in cols]

    loadings_df = pd.DataFrame(
        np.round(loadings, 3),
        index=friendly_feature_names,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    # Semantic axis interpretation heuristic
    pc1_top = loadings_df["PC1"].abs().idxmax()
    pc2_top = loadings_df["PC2"].abs().idxmax()

    axis_interpretations = {
        "PC1_name": f"PC1: {pc1_top} & Infrastructure Gradient ({explained_var[0]}% var)",
        "PC2_name": f"PC2: {pc2_top} & Environment Gradient ({explained_var[1]}% var)",
    }

    return {
        "pca_model": pca,
        "scaler": scaler,
        "coords_2d": coords_2d,
        "explained_variance_ratio": explained_var.tolist(),
        "total_explained_variance": float(cumulative_var),
        "loadings": loadings_df,
        "interpretations": axis_interpretations,
    }


def find_optimal_k(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    k_range: range = range(2, 7),
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates optimal number of clusters (k) using Silhouette Scores and Inertia (Elbow Method).
    Returns metrics table and best k recommendation.
    """
    cols = feature_cols or SCORE_COLUMNS
    X = df[cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []
    best_k = 3
    best_silhouette = -1.0

    for k in k_range:
        if k >= len(df):
            break
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        sil = float(np.round(silhouette_score(X_scaled, labels), 3))
        inertia = float(np.round(kmeans.inertia_, 2))

        results.append({
            "k": k,
            "silhouette_score": sil,
            "inertia": inertia,
        })

        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k

    eval_df = pd.DataFrame(results)

    return {
        "evaluation_table": eval_df,
        "best_k": best_k,
        "best_silhouette": best_silhouette,
    }


def generate_cluster_archetypes(
    centroids_df: pd.DataFrame,
) -> Dict[int, str]:
    """
    Assigns descriptive, human-interpretable archetypes to each cluster
    based on its centroid profile across the 7 livability dimensions.
    """
    archetypes: Dict[int, str] = {}

    for cluster_id, row in centroids_df.iterrows():
        # Identify standout strengths and characteristics
        top_pillar = row.idxmax()
        lowest_pillar = row.idxmin()

        conn = row.get("score_connectivity", 50.0)
        edu = row.get("score_education", 50.0)
        afford = row.get("score_affordability", 50.0)
        air = row.get("score_air_quality", 50.0)
        climate = row.get("score_climate", 50.0)
        health = row.get("score_healthcare", 50.0)
        safety = row.get("score_safety", 50.0)

        # Heuristic archetype classification
        if conn >= 75.0 and edu >= 75.0 and afford <= 50.0:
            name = "Tier 1 Metro Powerhouse"
        elif (air >= 75.0 or climate >= 80.0) and safety >= 70.0:
            name = "Wellness & Quality-of-Life Haven"
        elif afford >= 70.0 and (edu >= 50.0 or conn >= 50.0):
            name = "Affordable High-Growth Hub"
        elif health >= 75.0 and edu >= 70.0:
            name = "Healthcare & Education Center"
        elif afford >= 70.0 and conn < 55.0:
            name = "Developing Regional / Interior Belt"
        else:
            name = f"Balanced Growth Tier (Lead: {SCORE_COL_TO_PILLAR.get(top_pillar, top_pillar)})"

        archetypes[int(cluster_id)] = name

    # Ensure all cluster names are distinct
    seen = {}
    for cid, name in list(archetypes.items()):
        if name in seen:
            seen[name] += 1
            archetypes[cid] = f"{name} {seen[name]}"
        else:
            seen[name] = 1

    return archetypes


def run_clustering_analysis(
    df: pd.DataFrame,
    n_clusters: int = 4,
    feature_cols: Optional[List[str]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes K-Means clustering:
    - Standardizes features.
    - Fits KMeans with specified n_clusters.
    - Calculates overall Silhouette Score.
    - Generates cluster centroids table and descriptive archetype names.
    """
    cols = feature_cols or SCORE_COLUMNS
    X = df[cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = float(np.round(silhouette_score(X_scaled, labels), 3))

    # Calculate centroids in original 0-100 score space for interpretability
    df_temp = df[cols].copy()
    df_temp["cluster_id"] = labels
    centroids_df = df_temp.groupby("cluster_id").mean().round(2)

    # Generate archetype labels
    archetypes = generate_cluster_archetypes(centroids_df)

    # Map friendly column names for centroids
    friendly_centroids = centroids_df.rename(
        columns={c: SCORE_COL_TO_PILLAR.get(c, c) for c in cols}
    )
    friendly_centroids["Archetype"] = [archetypes[i] for i in friendly_centroids.index]
    friendly_centroids["City_Count"] = [(labels == i).sum() for i in friendly_centroids.index]

    return {
        "kmeans_model": kmeans,
        "labels": labels,
        "silhouette_score": sil_score,
        "archetypes": archetypes,
        "centroids": friendly_centroids,
    }


def run_full_ml_pipeline(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
    n_clusters: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the end-to-end ML pipeline:
    1. Evaluates optimal k (if n_clusters not explicitly given).
    2. Runs PCA and adds 'pca_x' and 'pca_y' coordinates to DataFrame.
    3. Runs K-Means and adds 'cluster_id' and 'cluster_label' to DataFrame.
    4. Returns enriched DataFrame and comprehensive ML diagnostic dictionary.
    """
    if df is None or df.empty:
        raise ValueError(f"Cannot run ML pipeline on empty DataFrame for {dataset_name}.")

    # 1. Optimal K evaluation
    k_eval = find_optimal_k(df, random_state=random_state)
    target_k = n_clusters or k_eval["best_k"]

    # Clamp k between 2 and len(df) - 1
    target_k = max(2, min(target_k, len(df) - 1))

    # 2. PCA
    pca_res = run_pca_analysis(df, random_state=random_state)

    # 3. K-Means
    kmeans_res = run_clustering_analysis(df, n_clusters=target_k, random_state=random_state)

    # 4. Enrich DataFrame
    enriched_df = df.copy()
    enriched_df["pca_x"] = np.round(pca_res["coords_2d"][:, 0], 3)
    enriched_df["pca_y"] = np.round(pca_res["coords_2d"][:, 1], 3)
    enriched_df["cluster_id"] = kmeans_res["labels"]
    enriched_df["cluster_label"] = enriched_df["cluster_id"].map(kmeans_res["archetypes"])

    ml_diagnostics = {
        "dataset_name": dataset_name,
        "k_evaluation": k_eval,
        "selected_k": target_k,
        "pca": pca_res,
        "kmeans": kmeans_res,
    }

    return enriched_df, ml_diagnostics


if __name__ == "__main__":
    print("=" * 75)
    print("  INDIA & KARNATAKA LIVABILITY INDEX - MODULE 4: ML ANALYTICS")
    print("=" * 75)

    # Load master processed datasets
    nat_csv = PROJECT_ROOT / "data" / "processed" / "master_national.csv"
    ka_csv = PROJECT_ROOT / "data" / "processed" / "master_karnataka.csv"

    if not nat_csv.exists() or not ka_csv.exists():
        from src.preprocessor import generate_master_datasets
        df_nat, df_ka = generate_master_datasets()
    else:
        df_nat = pd.read_csv(nat_csv)
        df_ka = pd.read_csv(ka_csv)

    # 1. Run Pan-India ML Pipeline (k=4)
    print("\n[1] Executing ML Pipeline on Pan-India Metros (k=4)...")
    enriched_nat, ml_nat = run_full_ml_pipeline(df_nat, dataset_name="Pan-India", n_clusters=4)
    print(f" -> PCA Explained Variance: PC1={ml_nat['pca']['explained_variance_ratio'][0]}%, PC2={ml_nat['pca']['explained_variance_ratio'][1]}% (Total: {ml_nat['pca']['total_explained_variance']}%)")
    print(f" -> K-Means Silhouette Score: {ml_nat['kmeans']['silhouette_score']}")
    print("\nPan-India Cluster Archetypes & Centroids:")
    print(ml_nat["kmeans"]["centroids"][["Archetype", "City_Count", "Connectivity", "Education", "Affordability", "Air Quality"]].to_string())

    print("\nSample City Cluster Assignments:")
    print(enriched_nat[["name", "state", "cluster_label", "pca_x", "pca_y"]].head(8).to_string(index=False))

    # 2. Run Karnataka ML Pipeline (k=4)
    print("\n" + "-" * 75)
    print("[2] Executing ML Pipeline on Karnataka State Districts (k=4)...")
    enriched_ka, ml_ka = run_full_ml_pipeline(df_ka, dataset_name="Karnataka", n_clusters=4)
    print(f" -> PCA Explained Variance: PC1={ml_ka['pca']['explained_variance_ratio'][0]}%, PC2={ml_ka['pca']['explained_variance_ratio'][1]}% (Total: {ml_ka['pca']['total_explained_variance']}%)")
    print(f" -> K-Means Silhouette Score: {ml_ka['kmeans']['silhouette_score']}")
    print("\nKarnataka Cluster Archetypes & Centroids:")
    print(ml_ka["kmeans"]["centroids"][["Archetype", "City_Count", "Healthcare", "Education", "Affordability", "Air Quality"]].to_string())

    print("\nSample Karnataka District Cluster Assignments:")
    print(enriched_ka[["name", "region", "cluster_label", "pca_x", "pca_y"]].head(8).to_string(index=False))

    print("\n" + "=" * 75)
    print("  MODULE 4 ML ENGINE: ALL PIPELINES COMPLETED SUCCESSFULLY [OK]")
    print("=" * 75)
