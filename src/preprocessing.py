"""
src/preprocessing.py
====================
Data Cleaning, Outlier Scrubbing, and Feature Engineering for CarbonSync.

Fulfills:
- Unit III: Missing value imputation (ffill + median/interpolation fallback).
- Unit III: IQR (1.5 x IQR) outlier scrubbing and bounding.
- Unit III: Cyclical hour encoding (sin / cos) preserving temporal continuity.
- Unit V: Feature standardization with scikit-learn StandardScaler.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from src.ingestion import EMISSION_FACTORS, load_raw_data
except ImportError:
    from ingestion import EMISSION_FACTORS, load_raw_data



def detect_and_impute_missing(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strategy: str = "interpolate",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Identifies and imputes missing (NaN) values across numerical telemetry series.
    
    Strategies:
    - 'interpolate': Time-series linear interpolation followed by forward/backward fill.
    - 'median': Column-wise median imputation.
    - 'mean': Column-wise mean imputation.
    
    Returns:
    - Imputed DataFrame copy.
    - Dictionary with counts of missing values resolved per column.
    """
    df_clean = df.copy()
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    missing_counts: Dict[str, int] = {}
    for col in columns:
        if col not in df_clean.columns:
            continue
        n_missing = int(df_clean[col].isna().sum())
        missing_counts[col] = n_missing

        if n_missing > 0:
            if strategy == "interpolate":
                df_clean[col] = df_clean[col].interpolate(method="linear").bfill().ffill()
            elif strategy == "median":
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif strategy == "mean":
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                raise ValueError(f"Unknown imputation strategy: {strategy}")

    return df_clean, missing_counts


def scrub_iqr_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    iqr_multiplier: float = 1.5,
    method: str = "clip",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Detects and scrubs statistical outliers using the standard 1.5 x IQR rule.
    
    Bounds:
        IQR = Q3 - Q1
        Lower Bound = Q1 - 1.5 * IQR
        Upper Bound = Q3 + 1.5 * IQR
        
    Methods:
    - 'clip': Caps values at the lower and upper bounds (preserves data length).
    - 'filter': Drops outlier rows.
    
    Returns:
    - Cleaned DataFrame.
    - Bounds and count report per column.
    """
    df_clean = df.copy()
    if columns is None:
        columns = ["temperature_c", "grid_load_mw"]

    outlier_report: Dict[str, Dict[str, float]] = {}

    for col in columns:
        if col not in df_clean.columns:
            continue
        q1 = float(df_clean[col].quantile(0.25))
        q3 = float(df_clean[col].quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        num_outliers = int(outliers_mask.sum())

        if method == "clip":
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == "filter":
            df_clean = df_clean[~outliers_mask].copy()

        outlier_report[col] = {
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "outliers_scrubbed": num_outliers,
        }

    return df_clean, outlier_report


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cyclical sine and cosine temporal encodings:
        hour_sin = sin(2 * pi * hour / 24)
        hour_cos = cos(2 * pi * hour / 24)
        
    Preserves continuous periodic distance between 23:00 and 00:00 (Unit III).
    """
    df_feat = df.copy()
    hour = df_feat["hour"].values
    df_feat["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df_feat["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    return df_feat


def scale_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardizes feature matrix to zero mean and unit variance using scikit-learn StandardScaler (Unit V).
    """
    X = df[feature_cols].values
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def run_preprocessing_pipeline(
    raw_df: Optional[pd.DataFrame] = None,
    output_path: str = "data/processed/grid_telemetry_clean.csv",
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Executes the end-to-end data cleaning and transformation pipeline:
    1. Loads raw telemetry.
    2. Imputes missing sensor values.
    3. Scrubs 1.5 x IQR outliers.
    4. Enriches with cyclical temporal features (hour_sin, hour_cos).
    5. Validates carbon intensity calculation integrity.
    6. Saves cleaned artifact to data/processed/.
    """
    if raw_df is None:
        raw_df = load_raw_data()

    # Step 1: Imputation
    df_imputed, missing_info = detect_and_impute_missing(
        raw_df, columns=["temperature_c", "gen_gas_mw"], strategy="interpolate"
    )

    # Step 2: IQR Outlier Scrubbing
    df_scrubbed, outlier_info = scrub_iqr_outliers(
        df_imputed, columns=["temperature_c", "grid_load_mw"], iqr_multiplier=1.5, method="clip"
    )

    # Step 3: Cyclical Temporal Encoding
    df_clean = add_cyclical_features(df_scrubbed)

    # Step 4: Re-verify generation sums and carbon intensity
    generation_sources = list(EMISSION_FACTORS.keys())
    total_gen = df_clean[generation_sources].sum(axis=1)
    df_clean["total_generation_mw"] = np.round(total_gen, 2)

    carbon_sum = sum(df_clean[source] * factor for source, factor in EMISSION_FACTORS.items())
    df_clean["carbon_intensity_gco2_per_kwh"] = np.round(carbon_sum / df_clean["total_generation_mw"], 2)

    # Step 5: Save processed dataset
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    summary_metadata = {
        "rows": len(df_clean),
        "columns": len(df_clean.columns),
        "missing_imputed": missing_info,
        "outliers_handled": outlier_info,
        "output_path": output_path,
    }

    return df_clean, summary_metadata


def load_clean_data(data_path: str = "data/processed/grid_telemetry_clean.csv") -> pd.DataFrame:
    """
    Loads clean processed dataset. Runs full pipeline if file is not yet cached on disk.
    """
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    df_clean, _ = run_preprocessing_pipeline(output_path=data_path)
    return df_clean


if __name__ == "__main__":
    print("Executing Preprocessing Pipeline...")
    df_clean, meta = run_preprocessing_pipeline()
    print(f"Cleaned dataset saved to: {meta['output_path']}")
    print(f"Imputed missing: {meta['missing_imputed']}")
    print(f"Scrubbed outliers: {meta['outliers_handled']}")
