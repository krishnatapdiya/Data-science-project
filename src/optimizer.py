"""
src/optimizer.py
================
Load-Shift Arbitrage Engine for Deferrable Energy Demands.

Mathematical Model:
For a deferrable compute/cooling/charging job of duration D hours and power draw W kW
across forecast horizon H:
    Score(t) = sum_{tau=t}^{t+D-1} W * [lambda * Tariff_tau + (1 - lambda) * CI_tau * 10^-3]
    t* = argmin_{t in [0, H-D]} Score(t)

Where:
- lambda in [0, 1] balances financial cost (lambda=1.0) vs. carbon abatement (lambda=0.0).
- Tariff_tau is the hourly commercial rate ($/kWh).
- CI_tau is the predicted marginal carbon intensity (gCO2/kWh).
- Multiplier 10^-3 normalizes gCO2 to kg CO2.
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Canonical Job Presets for Deferrable Commercial Loads
JOB_PRESETS: Dict[str, Dict[str, Any]] = {
    "AI Cluster Batch Compute": {
        "name": "AI Cluster Batch Compute",
        "power_kw": 250.0,
        "duration_hours": 6,
        "default_baseline_start": 14,  # Unmanaged start at 2 PM (peak)
        "description": "Large-scale GPU model pre-training / batch inference jobs.",
        "icon": "🤖",
    },
    "Commercial HVAC Pre-cooling": {
        "name": "Commercial HVAC Pre-cooling",
        "power_kw": 120.0,
        "duration_hours": 4,
        "default_baseline_start": 13,  # Afternoon peak start
        "description": "Thermal energy storage and building pre-cooling ahead of occupancy peaks.",
        "icon": "❄️",
    },
    "Campus Water Treatment Pump": {
        "name": "Campus Water Treatment Pump",
        "power_kw": 75.0,
        "duration_hours": 5,
        "default_baseline_start": 17,  # Evening peak start
        "description": "Agricultural reservoir pumping and municipal water filtration.",
        "icon": "💧",
    },
    "Commercial EV Fleet Charging": {
        "name": "Commercial EV Fleet Charging",
        "power_kw": 180.0,
        "duration_hours": 4,
        "default_baseline_start": 18,  # Fleet return hour
        "description": "Logistics delivery vans and shuttle bus overnight depot charging.",
        "icon": "⚡",
    },
}


def optimize_load_shift(
    df: pd.DataFrame,
    start_idx: int = 0,
    horizon_hours: int = 24,
    power_kw: float = 250.0,
    duration_hours: int = 6,
    baseline_start_hour: int = 14,
    lambda_weight: float = 0.5,
) -> Dict[str, Any]:
    """
    Executes the sliding-window load shift optimization over horizon H.
    
    Parameters:
    - df: Clean telemetry DataFrame with tariff_usd_per_kwh and carbon_intensity_gco2_per_kwh.
    - start_idx: Row offset in df for horizon start.
    - horizon_hours: Planning horizon (default: 24h).
    - power_kw: Constant power consumption of deferrable asset (kW).
    - duration_hours: Run length required to finish task (hours).
    - baseline_start_hour: Unoptimized scheduled start hour within horizon [0, horizon - duration].
    - lambda_weight: Cost vs. Carbon trade-off factor in [0.0, 1.0].
      (1.0 = Pure Cost Arbitrage, 0.0 = Pure Carbon Abatement).
    """
    # Slice horizon
    end_idx = min(start_idx + horizon_hours, len(df))
    sub_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    H = len(sub_df)

    if duration_hours > H:
        raise ValueError(f"Job duration ({duration_hours}h) cannot exceed horizon ({H}h).")

    # Tariff column selection (prefer tariff_inr_per_kwh, fallback to tariff_usd_per_kwh)
    tariff_col = "tariff_inr_per_kwh" if "tariff_inr_per_kwh" in sub_df.columns else "tariff_usd_per_kwh"
    tariffs = sub_df[tariff_col].values
    carbon_intensities = sub_df["carbon_intensity_gco2_per_kwh"].values
    hours = sub_df["hour"].values
    timestamps = sub_df["timestamp"].values

    max_start = H - duration_hours
    baseline_start = int(np.clip(baseline_start_hour, 0, max_start))

    scores = []
    window_metrics = []

    for t in range(max_start + 1):
        window_tariff = tariffs[t : t + duration_hours]
        window_ci = carbon_intensities[t : t + duration_hours]

        # Financial cost (₹) = sum(W * Tariff)
        total_cost = float(np.sum(power_kw * window_tariff))

        # Carbon footprint (kg CO2) = sum(W * CI * 10^-3)
        total_carbon_kg = float(np.sum(power_kw * window_ci * 1e-3))

        # Combined Arbitrage Objective Score
        score = lambda_weight * total_cost + (1.0 - lambda_weight) * total_carbon_kg

        scores.append(score)
        window_metrics.append(
            {
                "start_hour_offset": t,
                "start_clock_hour": int(hours[t]),
                "start_timestamp": str(timestamps[t]),
                "end_clock_hour": int(hours[t + duration_hours - 1]),
                "score": round(score, 3),
                "cost_inr": round(total_cost, 2),
                "cost_usd": round(total_cost, 2),  # alias
                "carbon_kg": round(total_carbon_kg, 2),
            }
        )

    # Optimal window argmin
    opt_idx = int(np.argmin(scores))
    opt_metric = window_metrics[opt_idx]
    baseline_metric = window_metrics[baseline_start]

    # Savings calculation
    cost_saved = baseline_metric["cost_inr"] - opt_metric["cost_inr"]
    cost_saved_pct = (cost_saved / baseline_metric["cost_inr"]) * 100.0 if baseline_metric["cost_inr"] > 0 else 0.0

    carbon_saved_kg = baseline_metric["carbon_kg"] - opt_metric["carbon_kg"]
    carbon_saved_pct = (carbon_saved_kg / baseline_metric["carbon_kg"]) * 100.0 if baseline_metric["carbon_kg"] > 0 else 0.0

    # Build timeline vectors for plotting
    baseline_load_profile = np.zeros(H)
    optimal_load_profile = np.zeros(H)
    baseline_load_profile[baseline_start : baseline_start + duration_hours] = power_kw
    optimal_load_profile[opt_idx : opt_idx + duration_hours] = power_kw

    timeline_df = pd.DataFrame(
        {
            "horizon_hour": list(range(H)),
            "clock_hour": hours,
            "timestamp": timestamps,
            "tariff_inr_per_kwh": tariffs,
            "tariff_usd_per_kwh": tariffs,  # alias
            "carbon_intensity_gco2": carbon_intensities,
            "baseline_load_kw": baseline_load_profile,
            "optimal_load_kw": optimal_load_profile,
        }
    )

    return {
        "job_spec": {
            "power_kw": power_kw,
            "duration_hours": duration_hours,
            "horizon_hours": H,
            "lambda_weight": lambda_weight,
        },
        "optimal_start_offset": opt_idx,
        "optimal_start_clock_hour": opt_metric["start_clock_hour"],
        "optimal_timestamp": opt_metric["start_timestamp"],
        "baseline_start_offset": baseline_start,
        "baseline_start_clock_hour": baseline_metric["start_clock_hour"],
        "baseline_timestamp": baseline_metric["start_timestamp"],
        "baseline_cost_inr": baseline_metric["cost_inr"],
        "optimal_cost_inr": opt_metric["cost_inr"],
        "cost_saved_inr": round(cost_saved, 2),
        "baseline_cost_usd": baseline_metric["cost_inr"],
        "optimal_cost_usd": opt_metric["cost_inr"],
        "cost_saved_usd": round(cost_saved, 2),
        "cost_saved_pct": round(cost_saved_pct, 2),
        "baseline_carbon_kg": baseline_metric["carbon_kg"],
        "optimal_carbon_kg": opt_metric["carbon_kg"],
        "carbon_saved_kg": round(carbon_saved_kg, 2),
        "carbon_saved_pct": round(carbon_saved_pct, 2),
        "all_windows": window_metrics,
        "scores_by_start": scores,
        "timeline_df": timeline_df,
    }


def optimize_flexible_schedule(
    df: pd.DataFrame,
    start_idx: int = 0,
    horizon_hours: int = 24,
    power_kw: float = 250.0,
    duration_hours: int = 6,
    earliest_start_hour: int = 0,
    deadline_hour: int = 24,
    allow_split: bool = False,
    objective_mode: str = "balanced",  # 'cost', 'carbon', 'balanced', 'stress_avoidance'
    lambda_weight: float = 0.5,
    battery_capacity_kwh: float = 0.0,
    battery_max_discharge_kw: float = 0.0,
    stress_penalty_inr_per_kwh: float = 3.50,
) -> Dict[str, Any]:
    """
    Solves the end-to-end operational dispatch problem with arbitrary constraints,
    split-shift flexibility, and Battery Energy Storage System (BESS) integration.

    Returns an actionable hour-by-hour dispatch timetable, financial & carbon balance sheet,
    and automation exports (CSV / JSON).
    """
    import json

    end_idx = min(start_idx + horizon_hours, len(df))
    sub_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    H = len(sub_df)

    # Validate constraint bounds
    earliest_h = max(0, min(earliest_start_hour, H - 1))
    deadline_h = max(earliest_h + 1, min(deadline_hour, H))
    available_hours = deadline_h - earliest_h

    if duration_hours > available_hours:
        duration_hours = available_hours

    tariff_col = "tariff_inr_per_kwh" if "tariff_inr_per_kwh" in sub_df.columns else "tariff_usd_per_kwh"
    tariffs = sub_df[tariff_col].values.astype(float)
    carbon_intensities = sub_df["carbon_intensity_gco2_per_kwh"].values.astype(float)
    stress_labels = sub_df.get("grid_stress_label", pd.Series(np.zeros(H))).values.astype(int)
    hours = sub_df["hour"].values.astype(int)
    timestamps = sub_df["timestamp"].values

    # Determine hourly penalty and normalized objective weights
    # Effective cost incorporates peak congestion surcharge if grid is stressed
    effective_tariffs = tariffs + (stress_labels * stress_penalty_inr_per_kwh)

    # Compute hourly composite scores for optimization
    if objective_mode == "cost":
        hourly_scores = effective_tariffs
    elif objective_mode == "carbon":
        hourly_scores = carbon_intensities
    elif objective_mode == "stress_avoidance":
        hourly_scores = effective_tariffs + (stress_labels * 100.0)
    else:  # balanced
        hourly_scores = (lambda_weight * effective_tariffs) + ((1.0 - lambda_weight) * carbon_intensities * 0.05)

    # 1. BASELINE UNMANAGED SCHEDULE (Naively runs continuously starting at earliest_h)
    baseline_active = np.zeros(H, dtype=bool)
    baseline_active[earliest_h : earliest_h + duration_hours] = True

    # 2. OPTIMIZED SCHEDULE SELECTION
    opt_active = np.zeros(H, dtype=bool)

    if allow_split:
        # User allows splitting the run across the best non-contiguous hours in the window
        window_indices = np.arange(earliest_h, deadline_h)
        sorted_indices = window_indices[np.argsort(hourly_scores[window_indices])]
        selected_hours = sorted_indices[:duration_hours]
        opt_active[selected_hours] = True
    else:
        # Contiguous block optimization within [earliest_h, deadline_h]
        max_start = deadline_h - duration_hours
        best_block_score = float("inf")
        best_start = earliest_h

        for start_t in range(earliest_h, max_start + 1):
            block_score = np.sum(hourly_scores[start_t : start_t + duration_hours])
            if block_score < best_block_score:
                best_block_score = block_score
                best_start = start_t
        opt_active[best_start : best_start + duration_hours] = True

    # 3. BATTERY PEAK-SHAVING SIMULATION (BESS)
    # If battery is available, discharge it during scheduled operating hours when grid is expensive/dirty
    battery_discharge_kw = np.zeros(H, dtype=float)
    rem_battery_kwh = float(battery_capacity_kwh)

    if battery_capacity_kwh > 0 and battery_max_discharge_kw > 0:
        # Prioritize battery discharge during scheduled hours with highest tariff/carbon
        active_hours = np.where(opt_active)[0]
        # Sort active hours descending by cost
        sorted_active_by_cost = active_hours[np.argsort(-effective_tariffs[active_hours])]
        for h_idx in sorted_active_by_cost:
            if rem_battery_kwh <= 0:
                break
            discharge_kw = min(power_kw, battery_max_discharge_kw, rem_battery_kwh)
            battery_discharge_kw[h_idx] = discharge_kw
            rem_battery_kwh -= discharge_kw

    # 4. COMPUTE HOURLY POWER, COSTS, AND EMISSIONS
    baseline_power = baseline_active.astype(float) * power_kw
    baseline_cost = np.sum(baseline_power * effective_tariffs)
    baseline_carbon = np.sum(baseline_power * carbon_intensities * 1e-3)
    baseline_stress_hours = int(np.sum(baseline_active & (stress_labels == 1)))

    opt_load_kw = opt_active.astype(float) * power_kw
    net_grid_import_kw = np.maximum(0.0, opt_load_kw - battery_discharge_kw)

    opt_cost = np.sum(net_grid_import_kw * effective_tariffs)
    opt_carbon = np.sum(net_grid_import_kw * carbon_intensities * 1e-3)
    opt_stress_hours = int(np.sum((net_grid_import_kw > 0) & (stress_labels == 1)))

    cost_saved = baseline_cost - opt_cost
    cost_saved_pct = (cost_saved / baseline_cost * 100.0) if baseline_cost > 0 else 0.0

    carbon_saved = baseline_carbon - opt_carbon
    carbon_saved_pct = (carbon_saved / baseline_carbon * 100.0) if baseline_carbon > 0 else 0.0

    # 5. BUILD ACTIONABLE HOUR-BY-HOUR DISPATCH TIMETABLE
    dispatch_rows = []
    for t in range(H):
        is_operating = bool(opt_active[t])
        is_stress = bool(stress_labels[t] == 1)
        batt_kw = float(battery_discharge_kw[t])
        grid_kw = float(net_grid_import_kw[t])

        # Action recommendation directive
        if is_operating and batt_kw >= power_kw:
            directive = "🔋 BATTERY DISPATCH (Zero Grid Draw)"
            status_tag = "ACTIVE (BATTERY)"
        elif is_operating and batt_kw > 0:
            directive = f"⚡ HYBRID RUN ({batt_kw:.0f}kW Batt + {grid_kw:.0f}kW Grid)"
            status_tag = "ACTIVE (HYBRID)"
        elif is_operating:
            directive = f"⚡ FULL RUN ({power_kw:.0f} kW from Grid)"
            status_tag = "ACTIVE (GRID)"
        elif is_stress:
            directive = "🚨 STANDBY / AVOID (High Grid Stress Peak)"
            status_tag = "IDLE (STRESS)"
        elif t < earliest_h or t >= deadline_h:
            directive = "⏸️ OUTSIDE WINDOW (Standby)"
            status_tag = "IDLE (OFF-WINDOW)"
        else:
            directive = "⏸️ PAUSED (Shifted to Cheaper Hour)"
            status_tag = "IDLE (SHIFTED)"

        h_cost = grid_kw * effective_tariffs[t]
        h_carbon = grid_kw * carbon_intensities[t] * 1e-3

        dispatch_rows.append(
            {
                "hour_offset": t,
                "clock_hour": int(hours[t]),
                "time_label": f"{int(hours[t]):02d}:00",
                "timestamp": str(timestamps[t]),
                "tariff_inr": round(float(tariffs[t]), 2),
                "carbon_intensity": round(float(carbon_intensities[t]), 1),
                "grid_stress": "ALERT" if is_stress else "NORMAL",
                "status": status_tag,
                "scheduled_load_kw": round(float(opt_load_kw[t]), 1),
                "battery_supply_kw": round(batt_kw, 1),
                "grid_import_kw": round(grid_kw, 1),
                "hourly_cost_inr": round(float(h_cost), 2),
                "hourly_carbon_kg": round(float(h_carbon), 2),
                "action_directive": directive,
            }
        )

    dispatch_df = pd.DataFrame(dispatch_rows)

    # Automation Exports
    csv_export = dispatch_df.to_csv(index=False)
    json_export = json.dumps(
        {
            "job_spec": {
                "power_kw": power_kw,
                "duration_hours": duration_hours,
                "earliest_start_hour": earliest_h,
                "deadline_hour": deadline_h,
                "allow_split": allow_split,
                "objective_mode": objective_mode,
                "battery_capacity_kwh": battery_capacity_kwh,
            },
            "summary_results": {
                "baseline_cost_inr": round(float(baseline_cost), 2),
                "optimized_cost_inr": round(float(opt_cost), 2),
                "net_savings_inr": round(float(cost_saved), 2),
                "savings_pct": round(float(cost_saved_pct), 1),
                "carbon_saved_kg": round(float(carbon_saved), 2),
                "carbon_saved_pct": round(float(carbon_saved_pct), 1),
                "stress_hours_avoided": baseline_stress_hours - opt_stress_hours,
            },
            "dispatch_schedule": dispatch_rows,
        },
        indent=2,
    )

    return {
        "job_spec": {
            "power_kw": power_kw,
            "duration_hours": duration_hours,
            "earliest_start_hour": earliest_h,
            "deadline_hour": deadline_h,
            "allow_split": allow_split,
            "objective_mode": objective_mode,
            "lambda_weight": lambda_weight,
            "battery_capacity_kwh": battery_capacity_kwh,
        },
        "baseline": {
            "cost_inr": round(float(baseline_cost), 2),
            "carbon_kg": round(float(baseline_carbon), 2),
            "stress_hours": baseline_stress_hours,
            "profile_kw": baseline_power,
        },
        "optimized": {
            "cost_inr": round(float(opt_cost), 2),
            "carbon_kg": round(float(opt_carbon), 2),
            "stress_hours": opt_stress_hours,
            "load_profile_kw": opt_load_kw,
            "battery_profile_kw": battery_discharge_kw,
            "grid_import_kw": net_grid_import_kw,
        },
        "savings": {
            "cost_saved_inr": round(float(cost_saved), 2),
            "cost_saved_pct": round(float(cost_saved_pct), 1),
            "carbon_saved_kg": round(float(carbon_saved), 2),
            "carbon_saved_pct": round(float(carbon_saved_pct), 1),
            "stress_hours_avoided": max(0, baseline_stress_hours - opt_stress_hours),
            "battery_used_kwh": round(float(np.sum(battery_discharge_kw)), 1),
        },
        "dispatch_df": dispatch_df,
        "csv_export": csv_export,
        "json_export": json_export,
    }


if __name__ == "__main__":
    from src.preprocessing import load_clean_data

    df = load_clean_data()
    # Test optimization for AI Batch job
    job = JOB_PRESETS["AI Cluster Batch Compute"]
    res = optimize_load_shift(
        df,
        start_idx=0,
        horizon_hours=24,
        power_kw=job["power_kw"],
        duration_hours=job["duration_hours"],
        baseline_start_hour=job["default_baseline_start"],
        lambda_weight=0.5,
    )

    print("--- Load-Shift Optimization Test Result ---")
    print(f"Baseline Start: Hour {res['baseline_start_clock_hour']}:00 -> Cost: ₹{res['baseline_cost_inr']:,.2f}, Carbon: {res['baseline_carbon_kg']} kg CO2")
    print(f"Optimal Start:  Hour {res['optimal_start_clock_hour']}:00 -> Cost: ₹{res['optimal_cost_inr']:,.2f}, Carbon: {res['optimal_carbon_kg']} kg CO2")
    print(f"Savings: ₹{res['cost_saved_inr']:,.2f} ({res['cost_saved_pct']}%) | Carbon Abated: {res['carbon_saved_kg']} kg ({res['carbon_saved_pct']}%)")

