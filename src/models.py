"""
src/models.py
=============
Predictive Modeling & Diagnostics for CarbonSync.

Fulfills:
- Unit I: Train/test splitting and train vs. validation loss curve tracking (overfitting control).
- Unit V: Binary classification for Peak Stress Alerts with Confusion Matrix, Accuracy, Precision, Recall.
- Unit VI: Simple Linear Regression and Multiple Linear Regression (MLR) with R2, RMSE, MAE.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MLR_FEATURES = [
    "grid_load_mw",
    "temperature_c",
    "gen_solar_mw",
    "gen_wind_mw",
    "hour_sin",
    "hour_cos",
]

CLASSIFIER_FEATURES = [
    "grid_load_mw",
    "temperature_c",
    "hour",
    "is_weekend",
    "gen_solar_mw",
    "gen_wind_mw",
    "hour_sin",
    "hour_cos",
]

__all__ = [
    "MLR_FEATURES",
    "CLASSIFIER_FEATURES",
    "train_evaluate_mlr",
    "compute_overfitting_curve",
    "train_evaluate_classifier",
    "predict_realtime_condition",
]


def train_evaluate_mlr(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains and evaluates Simple and Multiple Linear Regression (MLR) models.
    Target: carbon_intensity_gco2_per_kwh (Unit VI requirement).
    """
    X_mlr = df[MLR_FEATURES].values
    X_slr = df[["grid_load_mw"]].values
    y = df["carbon_intensity_gco2_per_kwh"].values

    # Train / Test split (Unit I requirement)
    X_tr_m, X_te_m, y_tr, y_te = train_test_split(X_mlr, y, test_size=test_size, random_state=random_state)
    X_tr_s, X_te_s, _, _ = train_test_split(X_slr, y, test_size=test_size, random_state=random_state)

    # 1. Simple Linear Regression (SLR: Load -> Carbon Intensity)
    slr_model = LinearRegression()
    slr_model.fit(X_tr_s, y_tr)
    y_pred_slr = slr_model.predict(X_te_s)

    r2_slr = r2_score(y_te, y_pred_slr)
    rmse_slr = float(np.sqrt(mean_squared_error(y_te, y_pred_slr)))
    mae_slr = mean_absolute_error(y_te, y_pred_slr)

    # 2. Multiple Linear Regression (MLR)
    mlr_model = LinearRegression()
    mlr_model.fit(X_tr_m, y_tr)
    y_pred_mlr = mlr_model.predict(X_te_m)

    r2_mlr = r2_score(y_te, y_pred_mlr)
    rmse_mlr = float(np.sqrt(mean_squared_error(y_te, y_pred_mlr)))
    mae_mlr = mean_absolute_error(y_te, y_pred_mlr)

    # MLR Coefficients
    coef_dict = {feat: round(float(coef), 4) for feat, coef in zip(MLR_FEATURES, mlr_model.coef_)}
    intercept = round(float(mlr_model.intercept_), 2)

    # Residuals
    residuals_mlr = y_te - y_pred_mlr

    return {
        "features": MLR_FEATURES,
        "mlr_model": mlr_model,
        "slr_model": slr_model,
        "test_size": test_size,
        "metrics_mlr": {
            "r2": round(r2_mlr, 4),
            "rmse": round(rmse_mlr, 3),
            "mae": round(mae_mlr, 3),
        },
        "metrics_slr": {
            "r2": round(r2_slr, 4),
            "rmse": round(rmse_slr, 3),
            "mae": round(mae_slr, 3),
        },
        "coefficients": coef_dict,
        "intercept": intercept,
        "y_true_test": y_te,
        "y_pred_test": y_pred_mlr,
        "residuals": residuals_mlr,
        "improvement_pct": round(((rmse_slr - rmse_mlr) / rmse_slr) * 100.0, 2),
    }


def compute_overfitting_curve(
    df: pd.DataFrame,
    max_degree: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Computes train vs. validation loss curves across model complexities (polynomial degrees 1 to max_degree).
    Demonstrates overfitting control (Unit I requirement).
    """
    # Use load and temperature to predict carbon intensity
    X = df[["grid_load_mw", "temperature_c"]].values
    y = df["carbon_intensity_gco2_per_kwh"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=random_state)

    train_rmses = []
    val_rmses = []
    degrees = list(range(1, max_degree + 1))

    for deg in degrees:
        pipe = make_pipeline(
            PolynomialFeatures(degree=deg, include_bias=False),
            StandardScaler(),
            LinearRegression(),
        )
        pipe.fit(X_train, y_train)

        y_tr_pred = pipe.predict(X_train)
        y_val_pred = pipe.predict(X_val)

        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_tr_pred)))
        val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))

        train_rmses.append(round(train_rmse, 3))
        val_rmses.append(round(val_rmse, 3))

    return {
        "degrees": degrees,
        "train_rmse": train_rmses,
        "val_rmse": val_rmses,
        "optimal_degree": degrees[int(np.argmin(val_rmses))],
    }


def train_evaluate_classifier(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains and evaluates binary classification models for Grid Stress Alerts (Unit V requirement).
    Outputs: Confusion Matrix, Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    X = df[CLASSIFIER_FEATURES].values
    y = df["grid_stress_label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Logistic Regression Classifier
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Feature Importance (Standardized Odds Ratio / Coefficients)
    importance_df = pd.DataFrame(
        {
            "Feature": CLASSIFIER_FEATURES,
            "Coefficient": [round(float(c), 4) for c in clf.coef_[0]],
            "Absolute_Impact": [round(abs(float(c)), 4) for c in clf.coef_[0]],
        }
    ).sort_values(by="Absolute_Impact", ascending=False)

    return {
        "model": clf,
        "scaler": scaler,
        "features": CLASSIFIER_FEATURES,
        "confusion_matrix": cm.tolist(),
        "breakdown": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "feature_importance": importance_df,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def predict_realtime_condition(
    mlr_model: Any,
    clf_model: Any,
    scaler: Any,
    grid_load_mw: float,
    temperature_c: float,
    hour: int,
    is_weekend: int = 0,
    gen_solar_mw: float = 0.0,
    gen_wind_mw: float = 0.0,
) -> Dict[str, Any]:
    """
    Inference tool: Predicts marginal carbon intensity (gCO2/kWh) and grid stress probability (%)
    for arbitrary user inputs in the 'What-If' interactive calculator.
    """
    # Cyclical hour encoding
    hour_sin = float(np.sin(2.0 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2.0 * np.pi * hour / 24.0))

    # 1. Carbon intensity prediction via MLR
    mlr_input = np.array([[grid_load_mw, temperature_c, gen_solar_mw, gen_wind_mw, hour_sin, hour_cos]])
    pred_ci = float(mlr_model.predict(mlr_input)[0])

    # 2. Grid stress alert prediction via Logistic Regression
    clf_input = np.array([[grid_load_mw, temperature_c, hour, is_weekend, gen_solar_mw, gen_wind_mw, hour_sin, hour_cos]])
    clf_input_scaled = scaler.transform(clf_input)
    pred_stress = int(clf_model.predict(clf_input_scaled)[0])
    stress_prob = float(clf_model.predict_proba(clf_input_scaled)[0][1])

    # Recommendation heuristic
    if pred_stress == 1 or stress_prob > 0.5:
        recommendation = "🚨 AVOID HEAVY LOADS: Grid is congested with high stress risk. Surcharge tariff applies."
        status_color = "red"
    elif pred_ci < 220:
        recommendation = "🟢 EXCELLENT GREEN WINDOW: High renewable penetration and clean energy. Ideal for compute/EV loads."
        status_color = "green"
    elif pred_ci < 350:
        recommendation = "🟡 MODERATE GRID STATE: Balanced dispatch. Suitable for standard operation."
        status_color = "amber"
    else:
        recommendation = "🟠 HIGH CARBON INTENSITY: Thermal baseload active. Consider deferring non-urgent jobs."
        status_color = "orange"

    return {
        "predicted_carbon_intensity": round(pred_ci, 2),
        "predicted_stress_label": pred_stress,
        "stress_probability_pct": round(stress_prob * 100.0, 1),
        "recommendation": recommendation,
        "status_color": status_color,
    }


if __name__ == "__main__":
    from src.preprocessing import load_clean_data

    df = load_clean_data()
    print("--- Multiple Linear Regression Evaluation ---")
    mlr_res = train_evaluate_mlr(df)
    print(f"SLR (Load -> CI): R2={mlr_res['metrics_slr']['r2']}, RMSE={mlr_res['metrics_slr']['rmse']}")
    print(f"MLR (All Feats):  R2={mlr_res['metrics_mlr']['r2']}, RMSE={mlr_res['metrics_mlr']['rmse']}")
    print(f"Coefficients: {mlr_res['coefficients']}")

    print("\n--- Overfitting Tracking Curve ---")
    curve = compute_overfitting_curve(df)
    for deg, tr, val in zip(curve["degrees"], curve["train_rmse"], curve["val_rmse"]):
        print(f"  Degree {deg}: Train RMSE={tr}, Val RMSE={val}")
    print(f"Optimal Complexity Degree: {curve['optimal_degree']}")

    print("\n--- Grid Stress Classification ---")
    clf_res = train_evaluate_classifier(df)
    print(f"Accuracy:  {clf_res['accuracy']*100:.2f}%")
    print(f"Precision: {clf_res['precision']*100:.2f}%")
    print(f"Recall:    {clf_res['recall']*100:.2f}%")
    print(f"F1-Score:  {clf_res['f1']*100:.2f}%")
    print(f"ROC-AUC:   {clf_res['roc_auc']:.4f}")
    print(f"Confusion Matrix: {clf_res['confusion_matrix']}")
