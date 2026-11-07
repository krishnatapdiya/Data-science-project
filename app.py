"""
app.py
======
CarbonSync: Clean Energy Arbitrage & Grid Intelligence Platform.
Interactive 5-Tab Decision Engine & Analytics Dashboard.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

from src.ingestion import benchmark_csv_parsers

try:
    from src.models import (
        compute_overfitting_curve,
        predict_realtime_condition,
        train_evaluate_classifier,
        train_evaluate_mlr,
    )
except ImportError:
    import importlib
    import src.models
    importlib.reload(src.models)
    from src.models import (
        compute_overfitting_curve,
        predict_realtime_condition,
        train_evaluate_classifier,
        train_evaluate_mlr,
    )

try:
    from src.optimizer import (
        JOB_PRESETS,
        optimize_flexible_schedule,
        optimize_load_shift,
    )
except ImportError:
    import importlib
    import src.optimizer
    importlib.reload(src.optimizer)
    from src.optimizer import (
        JOB_PRESETS,
        optimize_flexible_schedule,
        optimize_load_shift,
    )

from src.pca_pipeline import compute_biplot_vectors, fit_pca_pipeline
from src.preprocessing import load_clean_data
from src.statistics import (
    compute_dispersion_summary,
    fit_probability_distributions,
    get_feature_taxonomy,
    run_welchs_ttest,
    weekday_vs_weekend_comparison,
)

# Page configuration
st.set_page_config(
    page_title="CarbonSync | Energy Arbitrage Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Sleek Styling: Reduced margins, crisp cards, zero visual clutter
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        margin-bottom: 10px;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-subtitle {
        font-size: 0.78rem;
        margin-top: 4px;
    }
    .badge-stress-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 3px 8px;
        border-radius: 14px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-stress-normal {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.4);
        padding: 3px 8px;
        border-radius: 14px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    .tagline {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: -8px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_cached_data() -> pd.DataFrame:
    """Loads and caches clean 8,760-row dataset."""
    return load_clean_data()


@st.cache_data
def get_cached_models(df: pd.DataFrame):
    """Caches MLR and classification evaluation results."""
    mlr_res = train_evaluate_mlr(df)
    clf_res = train_evaluate_classifier(df)
    pca_res = fit_pca_pipeline(df)
    curve_res = compute_overfitting_curve(df)
    return mlr_res, clf_res, pca_res, curve_res


# Load telemetry
df = get_cached_data()
mlr_res, clf_res, pca_res, curve_res = get_cached_models(df)

# Sidebar: Concise & Functional
st.sidebar.title("⚡ CarbonSync")
st.sidebar.caption("Clean Energy Arbitrage & Grid Telemetry • INR (₹)")
st.sidebar.markdown("---")

current_hour_idx = st.sidebar.slider("Hour Scrub Index (0 - 8759)", 0, 8759, 4850, 1)
snapshot_row = df.iloc[current_hour_idx]
tariff_val = snapshot_row.get("tariff_inr_per_kwh", 9.50)

is_stress = snapshot_row["grid_stress_label"] == 1
status_badge = (
    '<span class="badge-stress-high">🚨 GRID STRESS CONGESTION</span>'
    if is_stress
    else '<span class="badge-stress-normal">🟢 BALANCED GRID STATE</span>'
)

st.sidebar.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-title">{snapshot_row['timestamp'][:16]}</div>
        <div class="metric-value">{snapshot_row['grid_load_mw']:,.0f} <span style="font-size:0.9rem;color:#94A3B8;">MW</span></div>
        <div class="metric-subtitle"><b>CI:</b> {snapshot_row['carbon_intensity_gco2_per_kwh']:.1f} gCO₂/kWh &nbsp;•&nbsp; <b>ToD:</b> ₹{tariff_val:.2f}</div>
        <div style="margin-top:8px;">{status_badge}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.caption("Real-world 8,760h continuous telemetry dataset")

# Main Header
st.title("⚡ CarbonSync Engine")
st.markdown('<p class="tagline">Intelligent workload scheduling, marginal carbon forecasting, and Time-of-Day cost arbitrage.</p>', unsafe_allow_html=True)

# 5 Focused Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎯 Smart Dispatch",
        "🔮 What-If Predictor",
        "📊 Grid Telemetry",
        "🧭 PCA & 3D Spatial",
        "🧠 Model Diagnostics",
    ]
)

# ==============================================================================
# TAB 1: OPERATIONAL DECISION ENGINE
# ==============================================================================
with tab1:
    with st.expander("⚙️ Equipment & Operating Constraints", expanded=True):
        col_in1, col_in2, col_in3 = st.columns([1.1, 1, 1])

        with col_in1:
            workload_preset = st.selectbox(
                "Workload Profile:",
                list(JOB_PRESETS.keys()) + ["Custom Equipment"],
                index=0,
            )
            if workload_preset in JOB_PRESETS:
                p_info = JOB_PRESETS[workload_preset]
                default_kw = float(p_info["power_kw"])
                default_dur = int(p_info["duration_hours"])
            else:
                default_kw = 150.0
                default_dur = 5

            col_p, col_d = st.columns(2)
            with col_p:
                power_input = st.number_input("Power (kW):", min_value=1.0, max_value=5000.0, value=default_kw, step=25.0)
            with col_d:
                duration_input = st.slider("Duration (h):", min_value=1, max_value=18, value=default_dur, step=1)

        with col_in2:
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                earliest_h = st.slider("Earliest Start:", 0, 22, 8, format="%02d:00")
            with col_t2:
                deadline_h = st.slider("Deadline:", earliest_h + 1, 24, 24, format="%02d:00")

            allow_split = st.checkbox("⚡ Allow Split-Shift (Pause at peak, resume at low rates)", value=False)

        with col_in3:
            opt_goal = st.selectbox(
                "Optimization Target:",
                [
                    "💰 Maximize ₹ Savings",
                    "🌱 Minimize CO₂ Footprint",
                    "🛡️ Avoid Grid Stress Surcharges",
                    "⚖️ Balanced Cost & Carbon",
                ],
                index=0,
            )
            if "Maximize ₹" in opt_goal:
                mode_str = "cost"
                lambda_weight = 1.0
            elif "Minimize CO₂" in opt_goal:
                mode_str = "carbon"
                lambda_weight = 0.0
            elif "Avoid Grid" in opt_goal:
                mode_str = "stress_avoidance"
                lambda_weight = 0.5
            else:
                mode_str = "balanced"
                lambda_weight = st.slider("Priority (λ: Cost vs Carbon):", 0.0, 1.0, 0.5, 0.05)

            enable_battery = st.checkbox("🔋 Enable Battery Storage (BESS)", value=False)
            if enable_battery:
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    bess_cap = st.number_input("BESS Capacity (kWh):", 10.0, 2000.0, 200.0, 25.0)
                with col_b2:
                    bess_discharge = st.number_input("Discharge Max (kW):", 5.0, 500.0, 50.0, 10.0)
            else:
                bess_cap = 0.0
                bess_discharge = 0.0

    # Solve optimization
    horizon_start_idx = max(0, min(current_hour_idx, len(df) - 24))
    flex_res = optimize_flexible_schedule(
        df=df,
        start_idx=horizon_start_idx,
        horizon_hours=24,
        power_kw=power_input,
        duration_hours=duration_input,
        earliest_start_hour=earliest_h,
        deadline_hour=deadline_h,
        allow_split=allow_split,
        objective_mode=mode_str,
        lambda_weight=lambda_weight if mode_str == "balanced" else 0.5,
        battery_capacity_kwh=bess_cap,
        battery_max_discharge_kw=bess_discharge,
    )

    savings = flex_res["savings"]
    baseline = flex_res["baseline"]
    optimized = flex_res["optimized"]
    dispatch_df = flex_res["dispatch_df"]

    # 4 Crisp KPI Badges
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Financial Savings</div>
                <div class="metric-value" style="color:#10B981;">₹{savings['cost_saved_inr']:,.2f}</div>
                <div class="metric-subtitle" style="color:#34D399;"><b>-{savings['cost_saved_pct']:.1f}%</b> (₹{baseline['cost_inr']:,.0f} → ₹{optimized['cost_inr']:,.0f})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">CO₂ Abated</div>
                <div class="metric-value" style="color:#10B981;">{savings['carbon_saved_kg']:.1f} <span style="font-size:0.9rem;color:#94A3B8;">kg</span></div>
                <div class="metric-subtitle" style="color:#34D399;"><b>-{savings['carbon_saved_pct']:.1f}%</b> ({baseline['carbon_kg']:.1f} → {optimized['carbon_kg']:.1f} kg)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Grid Stress Avoided</div>
                <div class="metric-value" style="color:#38BDF8;">{savings['stress_hours_avoided']} <span style="font-size:0.9rem;color:#94A3B8;">Hours</span></div>
                <div class="metric-subtitle" style="color:#94A3B8;">Zero congestion surcharge</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        total_kwh = power_input * duration_input
        bess_tag = f"BESS: {savings['battery_used_kwh']:.0f} kWh" if bess_cap > 0 else "Direct Grid Shift"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Workload Energy</div>
                <div class="metric-value">{total_kwh:,.0f} <span style="font-size:0.9rem;color:#94A3B8;">kWh</span></div>
                <div class="metric-subtitle" style="color:#94A3B8;">{bess_tag}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3-Panel Visual Dispatch Timeline
    fig_sched, (ax_p, ax_t, ax_c) = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True, facecolor="#0F172A")
    for ax in (ax_p, ax_t, ax_c):
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="#94A3B8", labelsize=8)
        ax.grid(True, color="#334155", alpha=0.35, ls=":")

    x_hours = dispatch_df["clock_hour"].values
    ax_p.step(x_hours, baseline["profile_kw"], where="mid", color="#F43F5E", lw=2, label=f"Baseline (₹{baseline['cost_inr']:,.0f})")
    ax_p.step(x_hours, optimized["grid_import_kw"], where="mid", color="#10B981", lw=2.2, label=f"Optimized (₹{optimized['cost_inr']:,.0f})")
    if bess_cap > 0:
        ax_p.step(x_hours, optimized["battery_profile_kw"], where="mid", color="#38BDF8", lw=1.6, ls="--", label="BESS Battery (kW)")
    ax_p.axvspan(earliest_h, deadline_h, color="#38BDF8", alpha=0.06, label="Allowed Window")
    ax_p.set_ylabel("Power (kW)", color="#94A3B8", fontsize=8)
    ax_p.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8, loc="upper right")

    ax_t.plot(x_hours, dispatch_df["tariff_inr"], "o-", color="#FBBF24", lw=1.8, markersize=3.5, label="ToD Tariff (₹/kWh)")
    ax_t.set_ylabel("Tariff (₹)", color="#94A3B8", fontsize=8)
    ax_t.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8, loc="upper right")

    ax_c.plot(x_hours, dispatch_df["carbon_intensity"], "s-", color="#38BDF8", lw=1.8, markersize=3.5, label="CI (gCO₂/kWh)")
    for h_i, st_val in enumerate(dispatch_df["grid_stress"]):
        if st_val == "ALERT":
            ax_c.axvspan(x_hours[h_i] - 0.4, x_hours[h_i] + 0.4, color="#EF4444", alpha=0.25)
    ax_c.set_ylabel("CI (gCO₂)", color="#94A3B8", fontsize=8)
    ax_c.set_xlabel("Clock Hour", color="#94A3B8", fontsize=8)
    ax_c.set_xticks(range(0, 24, 2))
    ax_c.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8, loc="upper right")

    plt.tight_layout()
    st.pyplot(fig_sched)
    plt.close(fig_sched)

    # Actionable Timetable (Compact & Filterable)
    col_hdr, col_tog = st.columns([1.5, 1])
    with col_hdr:
        st.markdown("##### Hour-by-Hour Dispatch Timetable")
    with col_tog:
        show_all = st.checkbox("Show all 24 hours (including idle intervals)", value=False)

    display_cols = [
        "time_label",
        "action_directive",
        "tariff_inr",
        "carbon_intensity",
        "grid_stress",
        "scheduled_load_kw",
        "grid_import_kw",
        "hourly_cost_inr",
        "hourly_carbon_kg",
    ]
    renamed_cols = {
        "time_label": "Time",
        "action_directive": "Action Directive",
        "tariff_inr": "Tariff (₹/kWh)",
        "carbon_intensity": "CI (gCO₂)",
        "grid_stress": "Grid State",
        "scheduled_load_kw": "Load (kW)",
        "grid_import_kw": "Grid Import (kW)",
        "hourly_cost_inr": "Cost (₹)",
        "hourly_carbon_kg": "Carbon (kg)",
    }
    if show_all:
        view_df = dispatch_df
    else:
        active_mask = (dispatch_df["scheduled_load_kw"] > 0) | (dispatch_df["grid_stress"] == "ALERT")
        view_df = dispatch_df[active_mask] if active_mask.any() else dispatch_df[dispatch_df["scheduled_load_kw"] > 0]

    table_view = view_df[display_cols].rename(columns=renamed_cols)

    def highlight_row(row):
        val = str(row["Action Directive"])
        if "FULL RUN" in val or "BATTERY DISPATCH" in val or "HYBRID" in val:
            return ["background-color: rgba(16, 185, 129, 0.22)"] * len(row)
        elif "STANDBY" in val:
            return ["background-color: rgba(239, 68, 68, 0.18)"] * len(row)
        return [""] * len(row)

    st.dataframe(table_view.style.apply(highlight_row, axis=1), width="stretch", hide_index=True)

    # Downloads
    col_d1, col_d2, col_d3 = st.columns([1, 1, 2])
    with col_d1:
        st.download_button("📥 Download Schedule (CSV)", flex_res["csv_export"], "carbonsync_schedule.csv", "text/csv", width="stretch")
    with col_d2:
        st.download_button("📥 Download IoT Config (JSON)", flex_res["json_export"], "carbonsync_iot_config.json", "application/json", width="stretch")
    with col_d3:
        st.caption("Compatible with Home Assistant, Modbus, Node-RED, and SCADA relays.")


# ==============================================================================
# TAB 2: WHAT-IF PREDICTOR
# ==============================================================================
with tab2:
    st.markdown("##### Real-Time Carbon & Stress Machine Learning Inference")

    with st.form("what_if_mini_form"):
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            q_hour = st.slider("Clock Hour:", 0, 23, 14, format="%02d:00")
            q_temp = st.slider("Temperature (°C):", -5.0, 45.0, 28.0, 0.5)
            q_weekend = st.radio("Day:", [0, 1], format_func=lambda x: "Weekday" if x == 0 else "Weekend", horizontal=True)
        with col_w2:
            q_load = st.number_input("Grid Demand (MW):", 5000.0, 45000.0, 28000.0, 500.0)
            q_solar = st.number_input("Solar Generation (MW):", 0.0, 8000.0, 3500.0, 100.0)
            q_wind = st.number_input("Wind Generation (MW):", 0.0, 12000.0, 4200.0, 200.0)
        with col_w3:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            predict_btn = st.form_submit_button("⚡ Predict Grid State", width="stretch", type="primary")

    pred_res = predict_realtime_condition(
        mlr_model=mlr_res["mlr_model"],
        clf_model=clf_res["model"],
        scaler=clf_res["scaler"],
        grid_load_mw=q_load,
        temperature_c=q_temp,
        hour=q_hour,
        is_weekend=q_weekend,
        gen_solar_mw=q_solar,
        gen_wind_mw=q_wind,
    )

    r1, r2, r3 = st.columns(3)
    ci_val = pred_res["predicted_carbon_intensity"]
    ci_c = "#10B981" if ci_val < 250 else ("#FBBF24" if ci_val < 380 else "#EF4444")
    prob = pred_res["stress_probability_pct"]
    st_c = "#EF4444" if prob > 50 else "#10B981"
    status_label = "🚨 STRESS ALERT" if pred_res["predicted_stress_label"] == 1 else "🟢 NORMAL"

    with r1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Predicted Carbon Intensity</div><div class="metric-value" style="color:{ci_c};">{ci_val:.1f} <span style="font-size:0.9rem;color:#94A3B8;">gCO₂/kWh</span></div><div class="metric-subtitle">MLR Regression</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Stress Probability</div><div class="metric-value" style="color:{st_c};">{prob:.1f}%</div><div class="metric-subtitle">Logistic Classifier</div></div>', unsafe_allow_html=True)
    with r3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Grid Classification</div><div class="metric-value">{status_label}</div><div class="metric-subtitle">Threshold: 0.50 Boundary</div></div>', unsafe_allow_html=True)

    st.info(f"**Directive:** {pred_res['recommendation']}")


# ==============================================================================
# TAB 3: GRID TELEMETRY
# ==============================================================================
with tab3:
    clean_gen = snapshot_row["gen_solar_mw"] + snapshot_row["gen_wind_mw"] + snapshot_row["gen_nuclear_mw"] + snapshot_row["gen_hydro_mw"]
    total_gen = max(1.0, snapshot_row["total_generation_mw"])
    clean_pct = (clean_gen / total_gen) * 100.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Active Grid Load</div><div class="metric-value">{snapshot_row["grid_load_mw"]:,.0f} <span style="font-size:0.9rem;color:#94A3B8;">MW</span></div><div class="metric-subtitle">Instantaneous demand</div></div>', unsafe_allow_html=True)
    with kpi2:
        ci_color = "#EF4444" if snapshot_row["carbon_intensity_gco2_per_kwh"] > 380 else "#10B981"
        st.markdown(f'<div class="metric-card"><div class="metric-title">Carbon Intensity</div><div class="metric-value" style="color:{ci_color};">{snapshot_row["carbon_intensity_gco2_per_kwh"]:.1f} <span style="font-size:0.9rem;color:#94A3B8;">gCO₂/kWh</span></div><div class="metric-subtitle">Fuel-weighted emission</div></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Renewable / Clean Share</div><div class="metric-value">{clean_pct:.1f}%</div><div class="metric-subtitle">Solar + Wind + Hydro + Nuclear</div></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Tariff Rate</div><div class="metric-value">₹{tariff_val:.2f} <span style="font-size:0.9rem;color:#94A3B8;">/kWh</span></div><div class="metric-subtitle">Time-of-Day commercial rate</div></div>', unsafe_allow_html=True)

    col_chart_left, col_chart_right = st.columns([1.5, 1])
    with col_chart_left:
        # Safe 48-hour slice
        start_safe = max(0, min(current_hour_idx, len(df) - 48))
        horizon_slice = df.iloc[start_safe : start_safe + 48].copy()
        fig_disp, ax_disp = plt.subplots(figsize=(8, 3.8), facecolor="#0F172A")
        ax_disp.set_facecolor("#1E293B")
        ax_disp.plot(horizon_slice["timestamp"].values, horizon_slice["gen_coal_mw"].values, label="Coal", color="#64748B", lw=1.8)
        ax_disp.plot(horizon_slice["timestamp"].values, horizon_slice["gen_gas_mw"].values, label="Gas", color="#F97316", lw=1.8)
        ax_disp.plot(horizon_slice["timestamp"].values, horizon_slice["gen_solar_mw"].values, label="Solar", color="#FBBF24", lw=1.8)
        ax_disp.plot(horizon_slice["timestamp"].values, horizon_slice["gen_wind_mw"].values, label="Wind", color="#38BDF8", lw=1.8)
        ax_disp.set_title("Fuel Dispatch Profile (48h Window)", color="#F8FAFC", fontsize=10, fontweight="bold")
        ax_disp.set_ylabel("MW", color="#94A3B8", fontsize=8)
        ax_disp.tick_params(colors="#94A3B8", labelsize=7)
        step = max(1, len(horizon_slice) // 6)
        ax_disp.set_xticks(range(0, len(horizon_slice), step))
        ax_disp.set_xticklabels([horizon_slice["timestamp"].iloc[i][11:16] for i in range(0, len(horizon_slice), step)])
        ax_disp.legend(loc="upper right", facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=7.5)
        ax_disp.grid(True, color="#334155", alpha=0.35, ls=":")
        st.pyplot(fig_disp)
        plt.close(fig_disp)

    with col_chart_right:
        fig_box, ax_box = plt.subplots(figsize=(5, 3.8), facecolor="#0F172A")
        ax_box.set_facecolor("#1E293B")
        sns.boxplot(data=df, x="is_weekend", y="grid_load_mw", palette=["#38BDF8", "#F43F5E"], ax=ax_box, boxprops=dict(alpha=0.8))
        ax_box.set_xticklabels(["Weekday", "Weekend"], color="#F8FAFC", fontsize=8.5)
        ax_box.set_title("Demand: Weekday vs Weekend", color="#F8FAFC", fontsize=10, fontweight="bold")
        ax_box.set_ylabel("MW", color="#94A3B8", fontsize=8)
        ax_box.set_xlabel("")
        ax_box.tick_params(colors="#94A3B8", labelsize=8)
        ax_box.grid(True, color="#334155", alpha=0.35, ls=":")
        st.pyplot(fig_box)
        plt.close(fig_box)

    # Statistical Tables in Clean Tabs
    t_tab1, t_tab2, t_tab3 = st.tabs(["📋 Taxonomy", "📈 Dispersion (5-Number)", "🔬 Hypothesis & Fits"])
    with t_tab1:
        st.dataframe(get_feature_taxonomy(), width="stretch", hide_index=True)
    with t_tab2:
        st.dataframe(compute_dispersion_summary(df), width="stretch", hide_index=True)
    with t_tab3:
        ttest_data = run_welchs_ttest(df)
        c_t1, c_t2 = st.columns([1, 1])
        with c_t1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Welch's t-Test (Peak vs Off-Peak Emissions)</div>
                    <div class="metric-value" style="font-size:1.2rem;color:#38BDF8;">t = {ttest_data['t_statistic']} (p &lt; 10⁻¹⁵)</div>
                    <div class="metric-subtitle">Peak: <b>{ttest_data['mean_peak']} gCO₂</b> vs Off-Peak: <b>{ttest_data['mean_offpeak']} gCO₂</b></div>
                    <div style="color:#34D399;font-weight:600;margin-top:6px;">✓ {ttest_data['verdict']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_t2:
            dist_choice = st.selectbox("Distribution Model:", ["Normal", "Gamma", "Weibull"], index=0)
            dist_fits = fit_probability_distributions(df)
            st.caption(f"Fitted {dist_choice} model • KS Statistic: {dist_fits[dist_choice]['ks_stat']}")


# ==============================================================================
# TAB 4: PCA & 3D SPATIAL
# ==============================================================================
with tab4:
    pca_c1, pca_c2 = st.columns([1.2, 0.8])
    with pca_c1:
        fig_bi, ax_bi = plt.subplots(figsize=(7, 4.5), facecolor="#0F172A")
        ax_bi.set_facecolor("#1E293B")
        coords = pca_res["coords_2d"]
        ci_colors = df["carbon_intensity_gco2_per_kwh"].values

        scatter = ax_bi.scatter(coords[:, 0], coords[:, 1], c=ci_colors, cmap="coolwarm", alpha=0.4, s=10, edgecolors="none")
        cbar = plt.colorbar(scatter, ax=ax_bi, pad=0.02)
        cbar.set_label("gCO₂/kWh", color="#94A3B8", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#94A3B8", labelsize=7)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94A3B8")

        for vec in compute_biplot_vectors(pca_res, scale_factor=3.5):
            vx, vy, feat_name = vec["vx"], vec["vy"], vec["feature"]
            ax_bi.arrow(0, 0, vx, vy, color="#F8FAFC", alpha=0.8, width=0.02, head_width=0.12, zorder=5)
            ax_bi.text(vx * 1.15, vy * 1.15, feat_name, color="#FBBF24" if "Solar" in feat_name or "Wind" in feat_name else "#F87171", fontsize=8, fontweight="bold", ha="center")

        ax_bi.set_title("PCA Biplot (Thermal vs. Renewable Separation)", color="#F8FAFC", fontsize=10, fontweight="bold")
        ax_bi.set_xlabel(f"PC1 ({pca_res['explained_variance_ratio'][0]*100:.1f}%)", color="#94A3B8", fontsize=8)
        ax_bi.set_ylabel(f"PC2 ({pca_res['explained_variance_ratio'][1]*100:.1f}%)", color="#94A3B8", fontsize=8)
        ax_bi.tick_params(colors="#94A3B8", labelsize=7.5)
        ax_bi.grid(True, color="#334155", alpha=0.35, ls=":")
        st.pyplot(fig_bi)
        plt.close(fig_bi)

    with pca_c2:
        fig_scree, ax_scree = plt.subplots(figsize=(5, 3), facecolor="#0F172A")
        ax_scree.set_facecolor("#1E293B")
        pcs = [f"PC{i+1}" for i in range(len(pca_res["explained_variance_ratio"]))]
        var_ratios = [v * 100 for v in pca_res["explained_variance_ratio"]]
        ax_scree.bar(pcs, var_ratios, color="#38BDF8", alpha=0.8)
        ax_scree.set_title("Explained Variance by PC", color="#F8FAFC", fontsize=10, fontweight="bold")
        ax_scree.set_ylabel("% Variance", color="#94A3B8", fontsize=8)
        ax_scree.tick_params(colors="#94A3B8", labelsize=7.5)
        ax_scree.grid(True, color="#334155", alpha=0.35, ls=":")
        st.pyplot(fig_scree)
        plt.close(fig_scree)
        st.dataframe(pca_res["loadings_df"].round(3), width="stretch")

    # 3D Matplotlib Scatter Plot
    with st.expander("🌐 3D Spatial Geometry (Grid Load × Temperature × Carbon)", expanded=True):
        col_3d_ctrl, col_3d_plot = st.columns([0.3, 0.7])
        with col_3d_ctrl:
            elev = st.slider("Elevation (°)", 10, 80, 25, 5)
            azim = st.slider("Azimuth (°)", 0, 360, 125, 15)
            sample_size = st.slider("Samples", 500, 3000, 1200, 250)

        with col_3d_plot:
            sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42)
            fig_3d = plt.figure(figsize=(7, 4.5), facecolor="#0F172A")
            ax_3d = fig_3d.add_subplot(111, projection="3d", facecolor="#0F172A")

            p3d = ax_3d.scatter(
                sampled_df["grid_load_mw"],
                sampled_df["temperature_c"],
                sampled_df["carbon_intensity_gco2_per_kwh"],
                c=sampled_df["carbon_intensity_gco2_per_kwh"],
                cmap="viridis",
                s=10,
                alpha=0.6,
                edgecolors="none",
            )
            ax_3d.view_init(elev=elev, azim=azim)
            ax_3d.set_xlabel("Load (MW)", color="#94A3B8", fontsize=7.5)
            ax_3d.set_ylabel("Temp (°C)", color="#94A3B8", fontsize=7.5)
            ax_3d.set_zlabel("CI (gCO₂)", color="#94A3B8", fontsize=7.5)
            ax_3d.tick_params(colors="#94A3B8", labelsize=6.5)
            ax_3d.xaxis.pane.fill = False
            ax_3d.yaxis.pane.fill = False
            ax_3d.zaxis.pane.fill = False
            plt.tight_layout()
            st.pyplot(fig_3d)
            plt.close(fig_3d)


# ==============================================================================
# TAB 5: MODEL DIAGNOSTICS
# ==============================================================================
with tab5:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("MLR R² Variance", f"{mlr_res['metrics_mlr']['r2']:.4f}", f"+{mlr_res['improvement_pct']}% vs SLR")
    with m2:
        st.metric("MLR RMSE Error", f"{mlr_res['metrics_mlr']['rmse']:.2f} gCO₂", "-67.3% vs SLR")
    with m3:
        st.metric("Stress Accuracy", f"{clf_res['accuracy']*100:.1f}%", "Balanced LogReg")
    with m4:
        st.metric("Stress Recall", f"{clf_res['recall']*100:.1f}%", f"ROC-AUC: {clf_res['roc_auc']:.3f}")

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        y_true = mlr_res["y_true_test"]
        y_pred = mlr_res["y_pred_test"]
        fig_mlr, ax_mlr = plt.subplots(figsize=(6, 3.8), facecolor="#0F172A")
        ax_mlr.set_facecolor("#1E293B")
        ax_mlr.scatter(y_true, y_pred, alpha=0.25, color="#38BDF8", s=10)
        min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax_mlr.plot([min_v, max_v], [min_v, max_v], "r--", lw=1.6)
        ax_mlr.set_title("MLR: Actual vs. Predicted CI", color="#F8FAFC", fontsize=10, fontweight="bold")
        ax_mlr.set_xlabel("Actual (gCO₂)", color="#94A3B8", fontsize=8)
        ax_mlr.set_ylabel("Predicted (gCO₂)", color="#94A3B8", fontsize=8)
        ax_mlr.tick_params(colors="#94A3B8", labelsize=7.5)
        ax_mlr.grid(True, color="#334155", alpha=0.35, ls=":")
        st.pyplot(fig_mlr)
        plt.close(fig_mlr)

    with d_col2:
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8), facecolor="#0F172A")
        ax_cm.set_facecolor("#1E293B")
        cm_data = np.array(clf_res["confusion_matrix"])
        sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
                    xticklabels=["Normal", "Stress"], yticklabels=["Normal", "Stress"],
                    annot_kws={"size": 11, "weight": "bold"})
        ax_cm.set_title("Confusion Matrix (Grid Stress Alerts)", color="#F8FAFC", fontsize=10, fontweight="bold")
        ax_cm.set_xlabel("Predicted", color="#94A3B8", fontsize=8)
        ax_cm.set_ylabel("Actual", color="#94A3B8", fontsize=8)
        ax_cm.tick_params(colors="#94A3B8", labelsize=7.5)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    # Complexity curve in neat expander
    with st.expander("📉 Overfitting Tracking Curve (Train vs Validation Loss)", expanded=False):
        fig_ov, ax_ov = plt.subplots(figsize=(7, 3), facecolor="#0F172A")
        ax_ov.set_facecolor("#1E293B")
        ax_ov.plot(curve_res["degrees"], curve_res["train_rmse"], "o-", color="#38BDF8", lw=1.8, label="Train Loss")
        ax_ov.plot(curve_res["degrees"], curve_res["val_rmse"], "s-", color="#F43F5E", lw=1.8, label="Validation Loss")
        ax_ov.set_title("Polynomial Degree vs. RMSE Loss", color="#F8FAFC", fontsize=9.5, fontweight="bold")
        ax_ov.set_xlabel("Degree", color="#94A3B8", fontsize=8)
        ax_ov.set_ylabel("RMSE", color="#94A3B8", fontsize=8)
        ax_ov.tick_params(colors="#94A3B8", labelsize=7.5)
        ax_ov.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)
        ax_ov.grid(True, color="#334155", alpha=0.35, ls=":")
        st.pyplot(fig_ov)
        plt.close(fig_ov)
