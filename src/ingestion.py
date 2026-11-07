"""
src/ingestion.py
================
Data Ingestion & Synthetic Telemetry Generator for CarbonSync.

Features:
- Physics-informed synthetic hourly grid telemetry generator (8,760 rows = 1 year).
- Mathematical calculation of marginal Carbon Intensity (gCO2/kWh).
- Time-of-Day (ToD) tariff structure and Grid Stress classification.
- Controlled injection of sensor dropouts and noise for cleaning verification.
- Pure Python CSV parser using built-in lists & dicts (no pandas).
- Benchmarking utility comparing pure Python parsing vs. Pandas read_csv (Unit IV).
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Core emission factors (gCO2 / kWh) from PROJECT_SPEC
EMISSION_FACTORS: Dict[str, float] = {
    "gen_coal_mw": 820.0,
    "gen_gas_mw": 490.0,
    "gen_biomass_mw": 230.0,
    "gen_solar_mw": 40.0,
    "gen_wind_mw": 12.0,
    "gen_nuclear_mw": 12.0,
    "gen_hydro_mw": 12.0,
}


def generate_synthetic_data(
    num_hours: int = 8760,
    start_date: str = "2024-01-01 00:00:00",
    random_seed: int = 42,
    inject_anomalies: bool = True,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Generates realistic 8,760-hour (1 year) grid telemetry dataset.
    
    Models:
    - Diurnal and seasonal temperature fluctuations.
    - Dual-peak daily grid demand (morning 09-11 and evening 18-21).
    - Thermal baseload (coal/nuclear) + peaker dispatch (gas).
    - Weather-correlated renewable generation (solar bell curve, wind variations).
    - Marginal carbon intensity calculation according to PROJECT_SPEC formula.
    - Commercial Time-of-Day (ToD) tariff tiers ($/kWh).
    """
    rng = np.random.default_rng(random_seed)
    base_time = datetime.fromisoformat(start_date)
    timestamps = [base_time + timedelta(hours=i) for i in range(num_hours)]

    hours = np.array([ts.hour for ts in timestamps], dtype=int)
    days_of_year = np.array([ts.timetuple().tm_yday for ts in timestamps], dtype=int)
    days_of_week = np.array([ts.weekday() for ts in timestamps], dtype=int)  # 0=Mon, 6=Sun
    is_weekend = (days_of_week >= 5).astype(int)

    # Seasons: Winter=0 (Dec-Feb), Spring=1 (Mar-May), Summer=2 (Jun-Aug), Fall=3 (Sep-Nov)
    month_array = np.array([ts.month for ts in timestamps], dtype=int)
    season_names = []
    for m in month_array:
        if m in (12, 1, 2):
            season_names.append("Winter")
        elif m in (3, 4, 5):
            season_names.append("Spring")
        elif m in (6, 7, 8):
            season_names.append("Summer")
        else:
            season_names.append("Fall")

    # 1. Temperature Model (Seasonal sine + Diurnal sine + noise)
    # Seasonal: peak in summer (day ~200), minimum in winter (day ~20)
    seasonal_temp = 18.0 + 12.0 * np.sin(2 * np.pi * (days_of_year - 105) / 365.25)
    # Diurnal: coolest at 5am, warmest at 3pm (15:00)
    diurnal_temp = 5.0 * np.sin(2 * np.pi * (hours - 9) / 24)
    temp_noise = rng.normal(0, 1.8, size=num_hours)
    temperature_c = seasonal_temp + diurnal_temp + temp_noise

    # 2. Grid Load Demand Model (MW)
    # Baseload ~ 22,000 MW. Dual-peak profile: morning bump (hour 9-11) and evening peak (hour 18-21).
    diurnal_load = (
        3500.0 * np.exp(-0.5 * ((hours - 10) / 2.0) ** 2)
        + 5800.0 * np.exp(-0.5 * ((hours - 19) / 2.2) ** 2)
        - 2200.0 * np.exp(-0.5 * ((hours - 3.5) / 2.5) ** 2)
    )
    # Weekend load drop (~15%)
    weekend_factor = np.where(is_weekend == 1, 0.85, 1.0)
    # Weather-driven HVAC load (cooling > 24C, heating < 10C)
    cooling_load = np.maximum(0.0, temperature_c - 22.0) * 280.0
    heating_load = np.maximum(0.0, 12.0 - temperature_c) * 190.0
    load_noise = rng.normal(0, 450.0, size=num_hours)

    grid_load_mw = (20000.0 + diurnal_load + cooling_load + heating_load) * weekend_factor + load_noise
    grid_load_mw = np.clip(grid_load_mw, 14000.0, 42000.0)

    # 3. Generation Fleet Profiles
    # Nuclear: Firm baseload ~ 4,200 MW with minor scheduled refueling dips
    gen_nuclear_mw = 4200.0 + 120.0 * np.sin(2 * np.pi * days_of_year / 365.25) + rng.normal(0, 30.0, size=num_hours)
    gen_nuclear_mw = np.clip(gen_nuclear_mw, 3800.0, 4500.0)

    # Biomass: Steady dispatch ~ 650 MW
    gen_biomass_mw = 650.0 + rng.normal(0, 25.0, size=num_hours)
    gen_biomass_mw = np.clip(gen_biomass_mw, 500.0, 800.0)

    # Solar: Bell curve during daylight (06:00 to 19:00), zero at night, higher in summer
    daylight_mask = (hours >= 6) & (hours <= 19)
    solar_zenith = np.maximum(0.0, np.sin(np.pi * (hours - 6) / 13.0))
    solar_seasonal = 0.75 + 0.35 * np.sin(2 * np.pi * (days_of_year - 80) / 365.25)
    cloud_cover = np.clip(rng.beta(2, 5, size=num_hours), 0.0, 0.8)
    gen_solar_mw = np.where(
        daylight_mask,
        6500.0 * solar_zenith * solar_seasonal * (1.0 - cloud_cover) + rng.normal(0, 50.0, size=num_hours),
        0.0,
    )
    gen_solar_mw = np.clip(gen_solar_mw, 0.0, 7500.0)

    # Wind: Stronger in winter/spring, slight nocturnal peak
    wind_diurnal = 500.0 * np.cos(2 * np.pi * (hours - 2) / 24)
    wind_seasonal = 1.0 + 0.3 * np.cos(2 * np.pi * (days_of_year - 40) / 365.25)
    wind_variability = rng.weibull(2.0, size=num_hours) * 1800.0
    gen_wind_mw = np.clip((2200.0 + wind_diurnal) * wind_seasonal + wind_variability, 400.0, 6800.0)

    # Hydro: Flexible hydro with seasonal spring run-off + peak load following
    hydro_base = 2000.0 + 600.0 * np.sin(2 * np.pi * (days_of_year - 60) / 365.25)
    hydro_dispatch = 0.08 * (grid_load_mw - 20000.0)
    gen_hydro_mw = np.clip(hydro_base + hydro_dispatch + rng.normal(0, 60.0, size=num_hours), 1200.0, 4000.0)

    # Thermal Dispatch to Balance Residual Load:
    # Residual Demand = Grid Load + 3% Transmission Losses - (Renewables + Nuclear + Biomass)
    target_generation = grid_load_mw * 1.03
    clean_and_firm = gen_nuclear_mw + gen_biomass_mw + gen_solar_mw + gen_wind_mw + gen_hydro_mw
    residual_demand = np.maximum(5000.0, target_generation - clean_and_firm)

    # Coal handles 65% of residual baseload up to capacity, Gas handles remainder peaker load
    gen_coal_mw = np.clip(0.68 * residual_demand + rng.normal(0, 150.0, size=num_hours), 6000.0, 18000.0)
    gas_needed = np.maximum(800.0, target_generation - (clean_and_firm + gen_coal_mw))
    gen_gas_mw = np.clip(gas_needed + rng.normal(0, 120.0, size=num_hours), 800.0, 12000.0)

    # 4. Marginal Carbon Intensity Calculation (gCO2 / kWh)
    total_generation_mw = (
        gen_coal_mw
        + gen_gas_mw
        + gen_biomass_mw
        + gen_solar_mw
        + gen_wind_mw
        + gen_nuclear_mw
        + gen_hydro_mw
    )

    carbon_emissions_sum = (
        gen_coal_mw * EMISSION_FACTORS["gen_coal_mw"]
        + gen_gas_mw * EMISSION_FACTORS["gen_gas_mw"]
        + gen_biomass_mw * EMISSION_FACTORS["gen_biomass_mw"]
        + gen_solar_mw * EMISSION_FACTORS["gen_solar_mw"]
        + gen_wind_mw * EMISSION_FACTORS["gen_wind_mw"]
        + gen_nuclear_mw * EMISSION_FACTORS["gen_nuclear_mw"]
        + gen_hydro_mw * EMISSION_FACTORS["gen_hydro_mw"]
    )
    carbon_intensity_gco2 = carbon_emissions_sum / total_generation_mw

    # 5. Commercial Time-of-Day (ToD) Tariff Structure in INR (₹/kWh)
    # Peak: 09:00-12:00 and 17:00-21:00 on weekdays = ₹14.50, weekends = ₹11.00
    # Off-peak: 22:00-06:00 = ₹6.50
    # Shoulder: remaining hours = ₹9.50
    is_peak_hour = np.zeros(num_hours, dtype=int)
    tariff_inr = np.full(num_hours, 9.50)

    for i in range(num_hours):
        h = hours[i]
        wk = is_weekend[i]
        if (9 <= h < 12) or (17 <= h < 21):
            if wk == 0:
                is_peak_hour[i] = 1
                tariff_inr[i] = 14.50
            else:
                tariff_inr[i] = 11.00
        elif (h >= 22) or (h < 6):
            tariff_inr[i] = 6.50
        else:
            tariff_inr[i] = 9.50

    # 6. Grid Stress Classification Label (Binary: 0 = Normal, 1 = Grid Stress Window)
    # Stress occurs when load > 80th percentile or carbon intensity is in top 20% during peak windows
    load_p80 = np.percentile(grid_load_mw, 80)
    ci_p75 = np.percentile(carbon_intensity_gco2, 75)
    grid_stress_label = (
        (grid_load_mw >= load_p80) | ((carbon_intensity_gco2 >= ci_p75) & (is_peak_hour == 1))
    ).astype(int)

    # 7. Construct Raw DataFrame
    df = pd.DataFrame(
        {
            "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
            "hour": hours,
            "day_of_week": days_of_week,
            "day_name": [ts.strftime("%A") for ts in timestamps],
            "is_weekend": is_weekend,
            "season": season_names,
            "temperature_c": np.round(temperature_c, 2),
            "grid_load_mw": np.round(grid_load_mw, 2),
            "gen_coal_mw": np.round(gen_coal_mw, 2),
            "gen_gas_mw": np.round(gen_gas_mw, 2),
            "gen_biomass_mw": np.round(gen_biomass_mw, 2),
            "gen_solar_mw": np.round(gen_solar_mw, 2),
            "gen_wind_mw": np.round(gen_wind_mw, 2),
            "gen_nuclear_mw": np.round(gen_nuclear_mw, 2),
            "gen_hydro_mw": np.round(gen_hydro_mw, 2),
            "total_generation_mw": np.round(total_generation_mw, 2),
            "carbon_intensity_gco2_per_kwh": np.round(carbon_intensity_gco2, 2),
            "tariff_inr_per_kwh": np.round(tariff_inr, 2),
            "tariff_usd_per_kwh": np.round(tariff_inr, 2),  # backward compatibility alias
            "is_peak_hour": is_peak_hour,
            "grid_stress_label": grid_stress_label,
        }
    )

    # 8. Controlled injection of missing values & sensor spike outliers for Unit III cleaning
    if inject_anomalies:
        # Inject ~1.2% missing values in temperature and gas generation
        nan_indices_temp = rng.choice(num_hours, size=int(num_hours * 0.012), replace=False)
        nan_indices_gas = rng.choice(num_hours, size=int(num_hours * 0.010), replace=False)
        df.loc[nan_indices_temp, "temperature_c"] = np.nan
        df.loc[nan_indices_gas, "gen_gas_mw"] = np.nan

        # Inject ~0.3% sensor spikes (outliers) in grid_load_mw and temperature_c
        spike_indices_load = rng.choice(num_hours, size=int(num_hours * 0.003), replace=False)
        spike_indices_temp = rng.choice(num_hours, size=int(num_hours * 0.003), replace=False)
        df.loc[spike_indices_load, "grid_load_mw"] = df.loc[spike_indices_load, "grid_load_mw"] * 1.85
        df.loc[spike_indices_temp, "temperature_c"] = df.loc[spike_indices_temp, "temperature_c"] + 35.0

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def parse_csv_pure_python(filepath: str) -> List[Dict[str, Any]]:
    """
    Custom CSV reader implemented with pure Python lists and dicts.
    Fulfills Unit IV syllabus requirement: Benchmarked against Pandas read_csv.
    Does NOT use Pandas or external libraries.
    """
    records: List[Dict[str, Any]] = []
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if not row:
                continue
            record: Dict[str, Any] = {}
            for col_name, val in zip(headers, row):
                val_str = val.strip()
                if val_str == "" or val_str.lower() == "nan":
                    record[col_name] = None
                else:
                    # Attempt integer conversion
                    try:
                        record[col_name] = int(val_str)
                    except ValueError:
                        # Attempt float conversion
                        try:
                            record[col_name] = float(val_str)
                        except ValueError:
                            # Keep as raw string
                            record[col_name] = val_str
            records.append(record)
    return records


def benchmark_csv_parsers(filepath: str) -> Dict[str, Any]:
    """
    Benchmarks pure Python CSV parsing vs. Pandas read_csv.
    Returns timings, row counts, and performance metrics (Unit IV).
    """
    # 1. Pure Python Benchmark
    t0 = time.perf_counter()
    py_records = parse_csv_pure_python(filepath)
    t_python = time.perf_counter() - t0

    # 2. Pandas Benchmark
    t1 = time.perf_counter()
    df_pd = pd.read_csv(filepath)
    t_pandas = time.perf_counter() - t1

    speedup = t_python / t_pandas if t_pandas > 0 else float("inf")

    return {
        "num_rows": len(py_records),
        "num_columns": len(py_records[0]) if py_records else 0,
        "pure_python_time_sec": round(t_python, 4),
        "pandas_time_sec": round(t_pandas, 4),
        "pandas_speedup_factor": round(speedup, 2),
        "python_sample_first_row": py_records[0] if py_records else {},
    }


def download_and_process_real_dataset(
    output_path: str = "data/raw/grid_telemetry_raw.csv",
    year: int = 2018,
    inject_anomalies: bool = True,
) -> pd.DataFrame:
    """
    Downloads and structures real-world hourly energy dispatch & weather telemetry:
    - Sourced from Open-Source Grid & Weather time-series records.
    - Aggregates actual thermal, gas peaker, nuclear, hydro, solar, and wind dispatch.
    - Merges with real hourly multi-city weather station temperatures.
    - Computes marginal Carbon Intensity (gCO2/kWh) and Commercial ToD Tariff in INR (₹/kWh).
    - Preserves 8,760 hourly records (1 complete year).
    """
    energy_raw_path = "data/raw/energy_dataset_raw.csv"
    weather_raw_path = "data/raw/weather_features_raw.csv"
    energy_url = "https://raw.githubusercontent.com/IBM/watsonx-ai-samples/master/cpd5.2/data/energy/energy_dataset.csv"
    weather_url = "https://raw.githubusercontent.com/igorkemyeni/Spain-Energy-Demand-Forecasting-Using-Deep-Learning/master/weather_features.csv"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Download if missing
    import subprocess
    import urllib.request

    if not os.path.exists(energy_raw_path):
        print(f"Downloading real energy dataset from {energy_url}...")
        try:
            subprocess.run(["curl.exe", "-s", "-L", "--max-time", "30", energy_url, "-o", energy_raw_path], check=True)
        except Exception:
            urllib.request.urlretrieve(energy_url, energy_raw_path)

    if not os.path.exists(weather_raw_path):
        print(f"Downloading real weather dataset from {weather_url}...")
        try:
            subprocess.run(["curl.exe", "-s", "-L", "--max-time", "45", weather_url, "-o", weather_raw_path], check=True)
        except Exception:
            urllib.request.urlretrieve(weather_url, weather_raw_path)

    print("Processing and merging real-world telemetry...")
    df_energy = pd.read_csv(energy_raw_path)
    df_weather = pd.read_csv(weather_raw_path)

    # Clean and match timestamps
    df_weather["time_clean"] = df_weather["dt_iso"].astype(str).str[:19]
    weather_hourly = df_weather.groupby("time_clean")["temp"].mean().reset_index()
    # Convert Kelvin to Celsius
    weather_hourly["temperature_c"] = weather_hourly["temp"] - 273.15

    df_merged = pd.merge(df_energy, weather_hourly, left_on="time", right_on="time_clean", how="inner")

    # Filter for the target continuous 1-year window (8,760 hours)
    start_ts = f"{year}-01-01 00:00:00"
    end_ts = f"{year}-12-31 23:00:00"
    df_year = df_merged[(df_merged["time"] >= start_ts) & (df_merged["time"] <= end_ts)].copy()

    if len(df_year) < 8760:
        # Fallback to first 8,760 rows if year slicing is partial
        df_year = df_merged.iloc[:8760].copy()

    df_year = df_year.iloc[:8760].reset_index(drop=True)
    dt_series = pd.to_datetime(df_year["time"])

    hours = dt_series.dt.hour.values
    days_of_week = dt_series.dt.weekday.values
    is_weekend = (days_of_week >= 5).astype(int)
    day_names = dt_series.dt.day_name().values

    # Seasons
    month_arr = dt_series.dt.month.values
    seasons = []
    for m in month_arr:
        if m in (12, 1, 2):
            seasons.append("Winter")
        elif m in (3, 4, 5):
            seasons.append("Spring")
        elif m in (6, 7, 8):
            seasons.append("Summer")
        else:
            seasons.append("Fall")

    # Real Generation Fleet Mapping
    gen_coal = (
        df_year["generation fossil brown coal/lignite"].fillna(0)
        + df_year["generation fossil hard coal"].fillna(0)
    ).values
    gen_gas = df_year["generation fossil gas"].fillna(0).values
    gen_biomass = df_year["generation biomass"].fillna(df_year["generation biomass"].median()).values
    gen_solar = df_year["generation solar"].fillna(0).values
    gen_wind = df_year["generation wind onshore"].fillna(0).values
    gen_nuclear = df_year["generation nuclear"].fillna(df_year["generation nuclear"].median()).values
    gen_hydro = (
        df_year["generation hydro water reservoir"].fillna(0)
        + df_year["generation hydro run-of-river and poundage"].fillna(0)
    ).values

    grid_load = df_year["total load actual"].fillna(df_year["total load actual"].median()).values
    temperature_c = df_year["temperature_c"].fillna(df_year["temperature_c"].median()).values

    # Total generation
    total_gen = gen_coal + gen_gas + gen_biomass + gen_solar + gen_wind + gen_nuclear + gen_hydro
    # Ensure no division by zero
    total_gen = np.maximum(total_gen, 1000.0)

    # Marginal Carbon Intensity (gCO2 / kWh)
    carbon_emissions = (
        gen_coal * EMISSION_FACTORS["gen_coal_mw"]
        + gen_gas * EMISSION_FACTORS["gen_gas_mw"]
        + gen_biomass * EMISSION_FACTORS["gen_biomass_mw"]
        + gen_solar * EMISSION_FACTORS["gen_solar_mw"]
        + gen_wind * EMISSION_FACTORS["gen_wind_mw"]
        + gen_nuclear * EMISSION_FACTORS["gen_nuclear_mw"]
        + gen_hydro * EMISSION_FACTORS["gen_hydro_mw"]
    )
    carbon_intensity_gco2 = carbon_emissions / total_gen

    # Commercial Time-of-Day (ToD) Tariff Structure in INR (₹/kWh)
    # Peak (09:00-12:00 & 17:00-21:00 on weekdays): ₹14.50/kWh, weekends: ₹11.00/kWh
    # Off-peak (22:00-06:00): ₹6.50/kWh
    # Normal / Shoulder: ₹9.50/kWh
    num_hours = len(df_year)
    is_peak_hour = np.zeros(num_hours, dtype=int)
    tariff_inr = np.full(num_hours, 9.50)

    for i in range(num_hours):
        h = hours[i]
        wk = is_weekend[i]
        if (9 <= h < 12) or (17 <= h < 21):
            if wk == 0:
                is_peak_hour[i] = 1
                tariff_inr[i] = 14.50
            else:
                tariff_inr[i] = 11.00
        elif (h >= 22) or (h < 6):
            tariff_inr[i] = 6.50
        else:
            tariff_inr[i] = 9.50

    # Grid Stress Label
    load_p80 = np.percentile(grid_load, 80)
    ci_p75 = np.percentile(carbon_intensity_gco2, 75)
    grid_stress_label = (
        (grid_load >= load_p80) | ((carbon_intensity_gco2 >= ci_p75) & (is_peak_hour == 1))
    ).astype(int)

    df_clean = pd.DataFrame(
        {
            "timestamp": df_year["time"].values,
            "hour": hours,
            "day_of_week": days_of_week,
            "day_name": day_names,
            "is_weekend": is_weekend,
            "season": seasons,
            "temperature_c": np.round(temperature_c, 2),
            "grid_load_mw": np.round(grid_load, 2),
            "gen_coal_mw": np.round(gen_coal, 2),
            "gen_gas_mw": np.round(gen_gas, 2),
            "gen_biomass_mw": np.round(gen_biomass, 2),
            "gen_solar_mw": np.round(gen_solar, 2),
            "gen_wind_mw": np.round(gen_wind, 2),
            "gen_nuclear_mw": np.round(gen_nuclear, 2),
            "gen_hydro_mw": np.round(gen_hydro, 2),
            "total_generation_mw": np.round(total_gen, 2),
            "carbon_intensity_gco2_per_kwh": np.round(carbon_intensity_gco2, 2),
            "tariff_inr_per_kwh": np.round(tariff_inr, 2),
            "tariff_usd_per_kwh": np.round(tariff_inr, 2),  # backward compatibility alias
            "is_peak_hour": is_peak_hour,
            "grid_stress_label": grid_stress_label,
        }
    )

    # Controlled injection of dropouts & outliers for Unit III cleaning
    if inject_anomalies:
        rng = np.random.default_rng(42)
        nan_indices_temp = rng.choice(num_hours, size=int(num_hours * 0.012), replace=False)
        nan_indices_gas = rng.choice(num_hours, size=int(num_hours * 0.010), replace=False)
        df_clean.loc[nan_indices_temp, "temperature_c"] = np.nan
        df_clean.loc[nan_indices_gas, "gen_gas_mw"] = np.nan

        spike_indices_load = rng.choice(num_hours, size=int(num_hours * 0.003), replace=False)
        spike_indices_temp = rng.choice(num_hours, size=int(num_hours * 0.003), replace=False)
        df_clean.loc[spike_indices_load, "grid_load_mw"] = df_clean.loc[spike_indices_load, "grid_load_mw"] * 1.85
        df_clean.loc[spike_indices_temp, "temperature_c"] = df_clean.loc[spike_indices_temp, "temperature_c"] + 35.0

    df_clean.to_csv(output_path, index=False)
    print(f"Successfully processed real-world dataset ({len(df_clean)} rows) into {output_path}")
    return df_clean


def load_raw_data(data_path: str = "data/raw/grid_telemetry_raw.csv") -> pd.DataFrame:
    """
    Ensures data availability out-of-the-box:
    If the CSV does not exist, checks for real-world dataset download or synthesizes clean data.
    """
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    try:
        return download_and_process_real_dataset(output_path=data_path)
    except Exception as e:
        print(f"Real data acquisition note: {e}. Falling back to physics-informed synthetic telemetry generator.")
        return generate_synthetic_data(output_path=data_path)


if __name__ == "__main__":
    raw_path = "data/raw/grid_telemetry_raw.csv"
    try:
        print("Acquiring real-world telemetry dataset...")
        df_raw = download_and_process_real_dataset(output_path=raw_path)
    except Exception as ex:
        print(f"Falling back to synthetic generator: {ex}")
        df_raw = generate_synthetic_data(output_path=raw_path)

    print(f"Dataset ready: {len(df_raw)} rows, {len(df_raw.columns)} columns.")

    print("\nBenchmarking CSV parsers (Unit IV)...")
    bench = benchmark_csv_parsers(raw_path)
    print(f"Pure Python: {bench['pure_python_time_sec']}s")
    print(f"Pandas read_csv: {bench['pandas_time_sec']}s")
    print(f"Speedup: {bench['pandas_speedup_factor']}x")

