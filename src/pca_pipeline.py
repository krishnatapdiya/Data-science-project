"""
src/pca_pipeline.py
===================
Principal Component Analysis (PCA) for Generation Fuel Mix Decomposition.

Fulfills:
- Unit IV: NumPy matrix operations and indexing.
- Unit IV: Reducing 7 fuel generation sources to 2 principal components.
- Unit IV: Scree plot data, explained variance ratio, and cumulative variance.
- Unit IV & VI: Generation of 2D coordinates and biplot vectors revealing 
  thermal baseload vs. renewable generation separation.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DEFAULT_FUEL_COLS = [
    "gen_coal_mw",
    "gen_gas_mw",
    "gen_biomass_mw",
    "gen_solar_mw",
    "gen_wind_mw",
    "gen_nuclear_mw",
    "gen_hydro_mw",
]

FRIENDLY_NAMES = {
    "gen_coal_mw": "Coal",
    "gen_gas_mw": "Gas (Peaker)",
    "gen_biomass_mw": "Biomass",
    "gen_solar_mw": "Solar PV",
    "gen_wind_mw": "Wind",
    "gen_nuclear_mw": "Nuclear",
    "gen_hydro_mw": "Hydro",
}


def fit_pca_pipeline(
    df: pd.DataFrame,
    fuel_cols: List[str] | None = None,
    n_components: int = 2,
) -> Dict[str, Any]:
    """
    Fits PCA on standardized generation sources and extracts:
    - Transformed 2D principal component coordinates (PC1, PC2)
    - PCA model object & fitted StandardScaler
    - Explained variance ratio per component & cumulative variance
    - Loadings matrix (eigenvectors) mapping fuels to components
    - NumPy matrix indexing verification for Unit IV syllabus
    """
    if fuel_cols is None:
        fuel_cols = DEFAULT_FUEL_COLS

    X_raw = df[fuel_cols].values  # NumPy ndarray

    # NumPy matrix indexing demonstration (Unit IV requirement)
    sample_first_row = X_raw[0, :]
    sample_submatrix = X_raw[:5, 1:4]
    row_sums = np.sum(X_raw, axis=1)

    # 1. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 2. Fit full PCA for scree analysis
    full_pca = PCA(n_components=len(fuel_cols))
    full_pca.fit(X_scaled)
    scree_explained_variance = full_pca.explained_variance_ratio_
    scree_cumulative_variance = np.cumsum(scree_explained_variance)

    # 3. Fit 2-component PCA for dimensionality reduction and biplot
    pca_2d = PCA(n_components=n_components, random_state=42)
    coords_2d = pca_2d.fit_transform(X_scaled)

    # 4. Extract Loadings (Eigenvectors scaled by sqrt(eigenvalues) or raw components)
    loadings = pca_2d.components_.T  # Shape: (n_features, 2)
    loadings_df = pd.DataFrame(
        loadings,
        index=[FRIENDLY_NAMES.get(c, c) for c in fuel_cols],
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    # Attach coordinates to DataFrame
    df_transformed = df.copy()
    for i in range(n_components):
        df_transformed[f"PC{i+1}"] = coords_2d[:, i]

    # Compute correlation between PC1/PC2 and Carbon Intensity
    corr_pc1_ci = float(np.corrcoef(coords_2d[:, 0], df["carbon_intensity_gco2_per_kwh"].values)[0, 1])
    corr_pc2_ci = float(np.corrcoef(coords_2d[:, 1], df["carbon_intensity_gco2_per_kwh"].values)[0, 1])

    # Interpret components
    # Coal and Gas typically have high positive/negative loadings on PC1 (Fossil Peakers vs Clean)
    interpretation = (
        f"PC1 accounts for {scree_explained_variance[0]*100:.1f}% of total generation variance "
        f"(strongly aligned with thermal fossil dispatch, correlation with CI = {corr_pc1_ci:.2f}). "
        f"PC2 explains {scree_explained_variance[1]*100:.1f}% of variance (dominated by solar/wind renewable fluctuations). "
        f"Together, the first 2 principal components preserve {scree_cumulative_variance[1]*100:.1f}% of information."
    )

    return {
        "df_transformed": df_transformed,
        "coords_2d": coords_2d,
        "pca_model": pca_2d,
        "full_pca_model": full_pca,
        "scaler": scaler,
        "fuel_cols": fuel_cols,
        "loadings_df": loadings_df,
        "explained_variance_ratio": scree_explained_variance.tolist(),
        "cumulative_variance_ratio": scree_cumulative_variance.tolist(),
        "corr_pc1_ci": round(corr_pc1_ci, 3),
        "corr_pc2_ci": round(corr_pc2_ci, 3),
        "interpretation": interpretation,
        "numpy_demo": {
            "matrix_shape": X_raw.shape,
            "sample_submatrix": sample_submatrix.tolist(),
            "first_row_generation": sample_first_row.tolist(),
        },
    }


def compute_biplot_vectors(pca_result: Dict[str, Any], scale_factor: float = 3.0) -> List[Dict[str, Any]]:
    """
    Computes 2D vector endpoints for PCA biplot arrows.
    """
    loadings = pca_result["pca_model"].components_  # (2, n_features)
    fuel_cols = pca_result["fuel_cols"]

    vectors = []
    for idx, col in enumerate(fuel_cols):
        vx = loadings[0, idx] * scale_factor
        vy = loadings[1, idx] * scale_factor
        vectors.append(
            {
                "feature": FRIENDLY_NAMES.get(col, col),
                "vx": round(float(vx), 3),
                "vy": round(float(vy), 3),
            }
        )
    return vectors


if __name__ == "__main__":
    from src.preprocessing import load_clean_data

    df = load_clean_data()
    res = fit_pca_pipeline(df)
    print("PCA Scree Explained Variance (First 4 PCs):")
    for i, var in enumerate(res["explained_variance_ratio"][:4]):
        print(f"  PC{i+1}: {var*100:.2f}% (Cumulative: {res['cumulative_variance_ratio'][i]*100:.2f}%)")

    print("\nLoadings Matrix:")
    print(res["loadings_df"])
    print(f"\n{res['interpretation']}")
