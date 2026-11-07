# PROJECT_SPEC: CarbonSync
> Predictive Grid Strain & Carbon Arbitrage Engine for Deferrable Energy Loads

---

### 1. Problem Statement & Core Value
Commercial facilities and campuses run heavy deferrable loads (batch AI/compute clusters, HVAC pre-cooling, water pumps, EV charging). Running these during peak grid strain triggers:
1. **High Carbon Emissions:** Utilities dispatch fossil peaker plants (diesel/gas), spiking emissions by up to 300%.
2. **Economic Surcharges:** Commercial Time-of-Day (ToD) tariffs impose 20% to 50% surcharges during peak hours.

**Solution:** CarbonSync forecasts marginal carbon intensity ($\text{gCO}_2/\text{kWh}$), classifies grid stress windows, and outputs the optimal time window to shift deferrable loads to minimize both cost and carbon footprint.

---

### 2. Syllabus Implementation Matrix

| Unit & Focus | Implementation Requirements |
| :--- | :--- |
| **Unit I: Stats & Overfitting** | Train/test data splitting, probability distribution fitting, train vs. validation loss curve tracking to demonstrate overfitting control. |
| **Unit II: Data Types & Stats** | Explicit taxonomy (Continuous, Discrete, Nominal, Ordinal, Binary). Mean, Median, Mode, Variance, Std Dev, and IQR dispersion analysis. |
| **Unit III: Cleaning & Prep** | Missing value imputation, IQR ($1.5 \times \text{IQR}$) outlier scrubbing, cyclical hour encoding ($\sin/\cos$), and Welch's $t$-test ($H_0: \mu_{\text{peak}} = \mu_{\text{off-peak}}$). |
| **Unit IV: Python Core & PCA** | Custom CSV reader using pure Python lists/dicts (benchmarked vs. Pandas `read_csv`), NumPy matrix indexing, and PCA reducing 6+ fuel sources to 2 principal components. |
| **Unit V: Pandas, EDA & Classification** | Pandas `groupby`/aggregations, feature scaling (`StandardScaler`), binary classification for Peak Stress Alerts evaluated via **Confusion Matrix, Accuracy, Precision, and Recall**. |
| **Unit VI: Visualizations & Regression** | 2D plots (Histograms, Boxplots, Seaborn Heatmap, Multi-line dispatch curves), **3D Scatter Plot** (Load vs. Temp vs. Carbon Intensity), Simple & **Multiple Linear Regression (MLR)** with $R^2$, RMSE, and MAE. |

---

### 3. Core Mathematical Models

1. **Carbon Intensity ($CI_t$ in $\text{gCO}_2/\text{kWh}$):**
   $$CI_t = \frac{\sum (P_{\text{source}} \times \alpha_{\text{source}})}{\sum P_{\text{source}}}$$
   *(Emission factors $\alpha$: Coal=820, Gas=490, Biomass=230, Solar=40, Wind=12, Nuclear/Hydro=12)*

2. **Cyclical Temporal Encoding:**
   $$h_{\sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad h_{\cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$

3. **Multiple Linear Regression (MLR):**
   $$\hat{CI}_t = \beta_0 + \beta_1(\text{Load}_t) + \beta_2(\text{Temp}_t) + \beta_3(\text{Solar}_t) + \beta_4(\text{Wind}_t) + \beta_5(h_{\sin}) + \beta_6(h_{\cos})$$

4. **Load-Shift Arbitrage Engine:**
   For a job of duration $D$ hours and power draw $W$ kW across horizon $H$:
   $$\text{Score}(t) = \sum_{\tau=t}^{t+D-1} W \cdot \big[\lambda \cdot \text{Tariff}_\tau + (1-\lambda) \cdot \hat{CI}_\tau \times 10^{-3}\big]$$
   $$t^* = \arg\min_{t \in [0, H-D]} \text{Score}(t)$$

---

### 4. Repository Structure

```text
carbonsync/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_types_and_stats.ipynb       # Units I & II
│   ├── 02_cleaning_and_python_core.ipynb   # Units III & IV
│   ├── 03_pca_and_classification.ipynb    # Units IV & V
│   └── 04_regression_and_3d_viz.ipynb      # Unit VI
├── src/
│   ├── __init__.py
│   ├── ingestion.py       # Realistic synthetic hourly generator + pure Python list/dict parser
│   ├── preprocessing.py   # Imputation, IQR outlier removal, cyclic encoding, scaling
│   ├── statistics.py      # Five-number summary, dispersion metrics, Welch's t-test
│   ├── pca_pipeline.py    # PCA fit, loadings, scree data, 2D coordinates
│   ├── models.py          # Multiple Linear Regression & Stress Alert Classifier (with metrics)
│   └── optimizer.py       # Load-shift search algorithm (cost & CO2 savings)
├── app.py                 # Multi-tab Streamlit dashboard
├── requirements.txt
└── README.md
```

---

### 5. Streamlit Dashboard Layout (`app.py`)

- **Tab 1: Telemetry & Statistical Profiling (Units I, II, III)**
  - KPI cards (Current Load, Carbon Intensity, Active Generation Mix, Peak Status).
  - Attribute taxonomy table (types & units).
  - Five-number summary tables & weekday vs. weekend boxplots.
  - $t$-test result card proving peak power is dirtier ($p$-value & verdict).
- **Tab 2: PCA & 3D Spatial Analytics (Units IV & VI)**
  - 2D PCA biplot showing thermal baseload vs. renewable separation.
  - Matplotlib 3D scatter plot: $X = \text{Grid Load}$, $Y = \text{Ambient Temperature}$, $Z = \text{Carbon Intensity}$.
- **Tab 3: Model Diagnostics & Evaluation (Units V & VI)**
  - MLR Actual vs. Predicted scatter plot with $45^\circ$ line, residual distribution, and $R^2$/RMSE/MAE metrics.
  - Classification Confusion Matrix heatmap, Precision, Recall, and Accuracy.
- **Tab 4: Load-Shift Arbitrage Sandbox (Decision Engine)**
  - Inputs: Job presets (AI Batch, HVAC Pre-cooling, Water Pump), Power draw (kW), Duration (hours), Cost vs. Carbon priority slider ($\lambda$).
  - Outputs: Baseline vs. Optimized schedule timeline, total monetary savings ($/₹), and total carbon emissions abated ($\text{kg CO}_2$).

---

### 6. Dependencies (`requirements.txt`)
```text
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.30.0
```

---

### 7. Verification Criteria
1. `src/ingestion.py` generates 8,760 hourly rows (1 year) of clean synthetic data out-of-the-box if no external CSV is present.
2. All 4 notebooks run sequentially from cell 1 to end without errors.
3. `streamlit run app.py` launches cleanly with functional sliders, rendering all 2D charts and the 3D Matplotlib plot.