import pandas as pd
import numpy as np
import geopandas as gpd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import os
import urllib.request

# ================================
# 1. SETUP
# ================================

MODEL_PATH = "../models/pm25_xgboost_research_model.pkl"
DATA_PATH = "../data/raw/global_data.csv"
SAT_DATA_PATH = "../data/derived/satellite_pm25_features.csv"
SHAPEFILE_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
SHAPEFILE_PATH = "ne_110m_admin_0_countries.zip"

def load_data_and_predict():
    # Attempt to load model
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Trying src/")
        try:
            model = joblib.load(f"src/{MODEL_PATH}")
        except FileNotFoundError:
            # Fallback to un-trained mapping if model doesn't exist
            model = None
            print("WARNING: Model not found. Will plot actual 2023 values instead of forecasting.")
    else:
        model = joblib.load(MODEL_PATH)
        
    try:
        df_ground = pd.read_csv('C:/Users/91860/Desktop/AeroGraphFed/data/raw/global_data.csv')
        df_sat = pd.read_csv('C:/Users/91860/Desktop/AeroGraphFed/data/derived/satellite_pm25_features.csv')
    except:
        df_ground = pd.read_csv(DATA_PATH)
        df_sat = pd.read_csv(SAT_DATA_PATH)
        
    # Standardize column names
    if 'Year' in df_ground.columns:
        df_ground = df_ground.rename(columns={'Year': 'year'})
    target_col = 'Population-Weighted PM2.5 [ug/m3]'
    if target_col in df_ground.columns:
        df_ground = df_ground.rename(columns={target_col: 'pm25'})
    if 'Total Population [million people]' in df_ground.columns:
        df_ground = df_ground.rename(columns={'Total Population [million people]': 'population'})

    # Merge ground and satellite
    if "Region" in df_ground.columns:
        df = df_ground.merge(df_sat, on=["Region", "year"], how="inner")
        df = df.rename(columns={'CIESIN_PM25': 'MODIS_AOD_PROXY'})
    else:
        df = df_ground.copy()

    # Engineer Features if model exists
    if model is not None:
        df = df.sort_values(by=["Region", "year"])
        df["pm25_lag1"] = df.groupby("Region")["pm25"].shift(1)
        df["pm25_lag2"] = df.groupby("Region")["pm25"].shift(2)
        df["pm25_roll3_mean"] = df.groupby("Region")["pm25"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df["pm25_roll3_std"] = df.groupby("Region")["pm25"].transform(lambda x: x.rolling(3, min_periods=2).std())
        df["pm25_growth"] = df.groupby("Region")["pm25"].pct_change()
        if "MODIS_AOD_PROXY" in df.columns:
            df["sat_pm25_growth"] = df.groupby("Region")["MODIS_AOD_PROXY"].pct_change()
        df["log_population"] = np.log1p(df["population"])
        
        # We will forecast for the VERY LATEST year in the dataset (e.g., 2023)
        latest_year = df['year'].max()
        df_latest = df[df['year'] == latest_year].copy()
        
        features = [
            "year", "log_population", "pm25_lag1", "pm25_lag2", 
            "pm25_roll3_mean", "pm25_roll3_std", "pm25_growth", 
            "MODIS_AOD_PROXY", "sat_pm25_growth"
        ]
        
        features = [f for f in features if f in df_latest.columns]
        
        # Predict
        print(f"Generating Global Predictions for year {latest_year}...")
        df_latest['Predicted_PM25'] = model.predict(df_latest[features])
        
    else:
        # Fallback to just mapping the most recent year if model is missing
        latest_year = df['year'].max()
        df_latest = df[df['year'] == latest_year].copy()
        df_latest['Predicted_PM25'] = df_latest['pm25']
        
    return df_latest, latest_year

# ================================
# 2. STATIC RESEARCH-GRADE MAP
# ================================
def plot_static_map(df_latest, latest_year):
    print("Generating High-Res Publication Map...")
    
    # Download map if missing
    if not os.path.exists(SHAPEFILE_PATH):
        print("Downloading Natural Earth boundaries...")
        urllib.request.urlretrieve(SHAPEFILE_URL, SHAPEFILE_PATH)
        
    world = gpd.read_file(SHAPEFILE_PATH)
    name_col = 'NAME' if 'NAME' in world.columns else 'ADMIN' if 'ADMIN' in world.columns else 'name'
    world['Region'] = world[name_col].str.strip()
    
    # Merge mapped shapes with predictions
    map_df = world.merge(df_latest, on="Region", how="left")
    
    # Create matplotlib figure with dark dashboard aesthetic
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), facecolor='#111111')
    ax.set_facecolor('#111111')
    
    # Plot missing countries
    world.plot(ax=ax, color='#333333', edgecolor='#555555', linewidth=0.5)
    
    # Plot mapped predictions with a fiery color map simulating pollution
    map_df.plot(
        column='Predicted_PM25', 
        ax=ax, 
        cmap='magma', 
        legend=True,
        legend_kwds={
            'label': f"Forecasted PM2.5 (μg/m³)",
            'orientation': "horizontal",
            'shrink': 0.5,
            'pad': 0.05
        },
        missing_kwds={'color': 'none'}
    )
    
    plt.title(f"Global AI PM2.5 Pollution Forecast ({latest_year})", 
              fontsize=24, color='white', pad=20, weight='bold')
    
    # Customizing the colorbar text to white
    cb = ax.get_figure().axes[1]
    cb.tick_params(colors='white', labelsize=12)
    cb.xaxis.label.set_color('white')
    cb.xaxis.label.set_fontsize(14)
    
    ax.axis('off')
    
    plt.tight_layout()
    os.makedirs('../images', exist_ok=True)
    out_path = f"../images/global_ai_pollution_forecast_{latest_year}.png"
    plt.savefig(out_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    print(f"-> Saved robust static map to {out_path}")

# ================================
# 3. INTERACTIVE HTML DASHBOARD MAP
# ================================
def generate_interactive_dashboard(df_latest, latest_year):
    print("Generating Interactive NASA-Style Dashboard Map (HTML)...")
    
    # Needs ISO-3 codes for Plotly by default, but we can use locationmode="country names"
    import warnings
    warnings.filterwarnings("ignore", message=".*locationmode.*")
    
    fig = px.choropleth(
        df_latest,
        locations="Region",
        locationmode="country names",
        color="Predicted_PM25",
        hover_name="Region",
        hover_data={"Region": False, "Predicted_PM25": ":.2f", "population": ":.1f", "MODIS_AOD_PROXY": ":.2f"},
        color_continuous_scale=px.colors.sequential.YlOrRd,
        title=f"🌍 Global Deep Learning PM2.5 Forecast ({latest_year})",
        labels={'Predicted_PM25': 'Predicted PM2.5 (μg/m³)', 'population': "Population (M)", 'MODIS_AOD_PROXY': "Satellite PM2.5"}
    )

    # NASA / advanced dark styling
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgba(255,255,255,0.2)",
            projection_type='natural earth',
            bgcolor='#1e1e1e',
            lakecolor='#1e1e1e',
            showocean=True,
            oceancolor='#121212',
            landcolor='#242424'
        ),
        paper_bgcolor='#121212',
        plot_bgcolor='#121212',
        font=dict(color='white', size=14),
        title=dict(x=0.5, font=dict(size=24)),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    image_file = f"../images/interactive_global_pollution_dashboard_{latest_year}.png"
    try:
        os.makedirs('../images', exist_ok=True)
        fig.write_image(image_file, width=1920, height=1080, scale=2)
        print(f"-> Saved Dashboard Image to {image_file}")
    except Exception as e:
        print(f"-> Could not save as image (requires kaleido: pip install kaleido).\nError: {e}")
        # Fallback to HTML if kaleido isn't installed
        html_file = f"../images/interactive_global_pollution_dashboard_{latest_year}.html"
        fig.write_html(html_file)
        print(f"-> Saved interactive dashboard to {html_file} instead.")

if __name__ == "__main__":
    df_latest, latest_year = load_data_and_predict()
    plot_static_map(df_latest, latest_year)
    generate_interactive_dashboard(df_latest, latest_year)
    print("\nAdvanced Mapping Pipeline Complete!")
