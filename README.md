# 🌍 AeroGraphFed: Global PM2.5 AI Forecasting & Intelligence

An advanced, research-grade Machine Learning pipeline and interactive dashboard for modeling, forecasting, and understanding global PM2.5 air pollution trends. 

This project fuses ground-truth pollution data with **CIESIN Satellite MODIS/MISR** proxy features to train robust predictive models, offering **Explainable AI (SHAP)** diagnostics and interactive **NASA-style global visualizations**.

---

## 🌟 Key Features

1. **Research-Grade Machine Learning Pipeline (`src/backend_model.py`)**
   - Integrates ground-level PM2.5 measurements with high-resolution satellite GeoTIFF data.
   - Utilizes `TimeSeriesSplit` cross-validation to prevent temporal data leakage.
   - Implements advanced `XGBoost` modeling with early stopping and feature dropout constraints to prevent overfitting.
   - Automatically generates global predictive features (Rolling Means, Lags, Demographic Growth rates).

2. **Explainable AI (XAI) Diagnostics**
   - Full integration with **SHAP** (SHapley Additive exPlanations) to transparently show how factors like population, historical pollution, and satellite AOD proxies influence the model's predictions.

3. **Global AI Pollution Forecasting Maps (`src/generate_forecast_maps.py`)**
   - Automatically renders the model's predictions into **High-Resolution Static Maps** using `geopandas` and `matplotlib`.
   - Generates interactive, NASA-style HTML dashboards using `plotly` for web-based dataset exploration.

4. **Interactive Streamlit Intelligence Dashboard (`dashboard/app.py`)**
   - A fully responsive, light-mode web interface to interact with the pollution dataset.
   - Includes real-time prediction simulators where you can tweak inputs (like satellite readings or population) to see how the XGBoost model reacts in real-time.
   - Built-in dynamic SHAP visualizers to explain the real-time simulation logic.

---

## 📁 Project Structure

```text
AeroGraphFed/
│
├── data/
│   ├── raw/                  # Ground truth global PM2.5 tracking data
│   └── derived/              # Extracted country-level satellite PM2.5 proxies
│
├── images/                   # Output directory for generated static maps, SHAP plots, and HTML Dashboards
│
├── dashboard/
│   └── app.py                # The interactive Streamlit Web Dashboard
│
├── models/                   # Pickled legacy/baseline models and the compiled pm25_xgboost_research_model.pkl
│
├── src/                      
│   ├── extract_satellite_features.py  # Pipeline to extract CIESIN GeoTIFFs to tabular data using Geopandas
│   ├── backend_model.py               # The core XGBoost Training Pipeline & Validation
│   └── generate_forecast_maps.py      # Map generation script for static and dynamic plots
│
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

You need Python 3.9+ installed. Install the required spatial and machine learning dependencies:

```bash
pip install pandas numpy xgboost scikit-learn shap plotly geopandas rasterstats streamlit matplotlib joblib kaleido
```

### 1. Extract Satellite Features (Optional)
If you have raw CIESIN GeoTIFF files, extract them to country-level tabular bounds:
```bash
python src/extract_satellite_features.py
```
*(Note: requires the CIESIN dataset downloaded locally).*

### 2. Train the Research Model
Run the core training pipeline. This will perform Time-Series Cross Validation, train the final robust model, and output SHAP summary plots in the `src/` directory.
```bash
python src/backend_model.py
```

### 3. Generate Global Maps
Create the high-resolution publication maps and interactive HTML plots based on the latest model forecasts:
```bash
python src/generate_forecast_maps.py
```

### 4. Launch the AI Dashboard
Start the intelligent web interface to analyze trends, run live simulations, and interpret the model using SHAP:
```bash
streamlit run dashboard/app.py
```
---

## 📊 Data Sources

- **Ground PM2.5 & Population**: Curated historical global population-weighted datasets.
- **Satellite PM2.5 Proxy**: [CIESIN SEDAC Global Annual PM2.5 Grids from MODIS, MISR and SeaWiFS Aerosol Optical Depth (AOD)](https://sedac.ciesin.columbia.edu/data/set/sdei-global-annual-gwr-pm2-5-modis-misr-seawifs-aod-v4-gl-03).
- **Cartography**: `NaturalEarth` datasets used for spatial rendering and zonal boundary extraction.
