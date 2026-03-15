import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Global PM2.5 AI Dashboard", layout="wide", page_icon="🌍")

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
project_root = Path(__file__).resolve().parent.parent

data_path = project_root / "data" / "raw" / "global_data.csv"
sat_data_path = project_root / "data" / "derived" / "satellite_pm25_features.csv"
model_path = project_root / "models" / "pm25_xgboost_research_model.pkl"
features_path = project_root / "models" / "research_model_features.txt"

# ---------------------------------------------------
# CACHED DATA LOADING
# ---------------------------------------------------
@st.cache_data
def load_and_merge_data():
    try:
        df_ground = pd.read_csv(data_path)
    except:
        st.error(f"Ground truth dataset not found at {data_path}")
        st.stop()

    try:
        df_sat = pd.read_csv(sat_data_path)
    except:
        st.warning(f"Satellite PM2.5 Features missing from {sat_data_path}. Continuing with ground data only.")
        df_sat = pd.DataFrame()
        
    if 'Year' in df_ground.columns:
        df_ground = df_ground.rename(columns={'Year': 'year'})
    target_col = 'Population-Weighted PM2.5 [ug/m3]'
    if target_col in df_ground.columns:
        df_ground = df_ground.rename(columns={target_col: 'pm25'})
    if 'Total Population [million people]' in df_ground.columns:
        df_ground = df_ground.rename(columns={'Total Population [million people]': 'population'})

    if "Region" in df_ground.columns and not df_sat.empty:
        df = df_ground.merge(df_sat, on=["Region", "year"], how="left")
        df = df.rename(columns={'CIESIN_PM25': 'MODIS_AOD_PROXY'})
    else:
        df = df_ground.copy()
        if "MODIS_AOD_PROXY" not in df.columns:
            df["MODIS_AOD_PROXY"] = np.nan
        
    df = df.sort_values(by=["Region", "year"] if "Region" in df.columns else ["year"])
    
    # Simple feature engineering for dashboard metrics
    df["pm25_lag1"] = df.groupby("Region")["pm25"].shift(1)
    df["pm25_lag2"] = df.groupby("Region")["pm25"].shift(2)
    df["pm25_roll3_mean"] = df.groupby("Region")["pm25"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["pm25_roll3_std"] = df.groupby("Region")["pm25"].transform(lambda x: x.rolling(3, min_periods=2).std())
    df["pm25_growth"] = df.groupby("Region")["pm25"].pct_change()
    df["sat_pm25_growth"] = df.groupby("Region")["MODIS_AOD_PROXY"].pct_change()
    df["log_population"] = np.log1p(df["population"])
    
    return df

with st.spinner("Loading Research-Grade Datasets..."):
    df = load_and_merge_data()

# ---------------------------------------------------
# CACHED MODEL LOADING
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model = None
    feature_names = []

    if model_path.exists():
        model = joblib.load(model_path)
    else:
        st.warning(f"Research Model not found at {model_path}. Trying fallback...")
        fallback = project_root / "models" / "xgb_pm25_model.pkl"
        if fallback.exists():
            model = joblib.load(fallback)

    if features_path.exists():
        with open(features_path) as f:
            feature_names = [line.strip() for line in f.readlines()]
    else:
        fallback_feat = project_root / "models" / "dashboard_features.txt"
        if fallback_feat.exists():
             with open(fallback_feat) as f:
                 feature_names = [line.strip() for line in f.readlines()]

    return model, feature_names

model, feature_names = load_model()

# ---------------------------------------------------
# APP HEADER
# ---------------------------------------------------
st.title("🌍 Global PM2.5 Air Quality Intelligence Dashboard")
st.markdown("""
Welcome to the Global AI Pollution Dashboard.  
This dashboard integrates ground-based measurements with **MODIS/MISR Satellite derived AOD features** 
to accurately forecast regional PM2.5 and expose relationships via Explainable AI.
""")

# ---------------------------------------------------
# GLOBAL FORECAST MAP (NASA STYLE)
# ---------------------------------------------------
st.subheader("🛰️ Global Deep Learning PM2.5 Forecast Map")

col_map1, col_map2 = st.columns([1, 2])
with col_map1:
    latest_year = int(df['year'].max())
    selected_year_map = st.slider("Select Forecast Year to Render", int(df['year'].min()), latest_year, latest_year)
with col_map2:
    map_type = st.radio("Map Style", ["Choropleth (Pollution Intensity)", "Bubble Map (Population Risk)"], horizontal=True)

df_map = df[df['year'] == selected_year_map].copy()

if model is not None and feature_names:
    missing_cols = [f for f in feature_names if f not in df_map.columns]
    if not missing_cols:
        df_map_valid = df_map.dropna(subset=feature_names)
        if not df_map_valid.empty:
            df_map.loc[df_map_valid.index, 'Predicted_PM25'] = model.predict(df_map_valid[feature_names])
        else:
            df_map['Predicted_PM25'] = df_map['pm25']
    else:
        df_map['Predicted_PM25'] = df_map['pm25']
else:
    df_map['Predicted_PM25'] = df_map['pm25']

if map_type == "Choropleth (Pollution Intensity)":
    fig_map = px.choropleth(
        df_map,
        locations="Region",
        locationmode="country names",
        color="Predicted_PM25",
        hover_name="Region",
        hover_data={"Region": False, "Predicted_PM25": ":.2f", "population": ":.1f", "MODIS_AOD_PROXY": ":.2f"},
        color_continuous_scale=px.colors.sequential.YlOrRd,
        title=f"Predicted PM2.5 Levels ({selected_year_map})"
    )
else:
    fig_map = px.scatter_geo(
        df_map,
        locations="Region",
        locationmode="country names",
        color="Predicted_PM25",
        size="population",
        hover_name="Region",
        hover_data={"Region": False, "Predicted_PM25": ":.2f", "population": ":.1f", "MODIS_AOD_PROXY": ":.2f"},
        color_continuous_scale=px.colors.sequential.YlOrRd,
        title=f"Population Risk: PM2.5 Levels & Demographics ({selected_year_map})",
        size_max=40
    )

fig_map.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="rgba(0,0,0,0.2)",
        projection_type='natural earth',
        bgcolor='white',
        lakecolor='white',
        showocean=True,
        oceancolor='#e0f3f8',
        landcolor='#f7f7f7'
    ),
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(color='black', size=14),
    margin=dict(l=0, r=0, t=40, b=0),
    height=600
)

st.plotly_chart(fig_map, use_container_width=True)

# ---------------------------------------------------
# GLOBAL TREND FORECAST (PROPHET ML)
# ---------------------------------------------------
st.subheader("📈 Global ML Pollution Trend Forecast (Next 10 Years)")
with st.spinner("Generating Time-Series Forecast..."):
    try:
        from prophet import Prophet
        import logging
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
        
        # Calculate global annual average for Prophet
        trend_df = df.groupby("year")["pm25"].mean().reset_index()
        prophet_df = trend_df.rename(columns={"year": "ds", "pm25": "y"})
        # Prophet expects datetime, so convert years to Jan 1st of that year
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'].astype(int), format='%Y')
        
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        m.fit(prophet_df)
        
        future = m.make_future_dataframe(periods=10, freq="YS")
        forecast = m.predict(future)
        
        fig_forecast = go.Figure()
        
        # Plot historical with a smooth gradient
        fig_forecast.add_trace(go.Scatter(
            x=prophet_df['ds'].dt.year, 
            y=prophet_df['y'], 
            mode='lines+markers', 
            name='Historical Global Average',
            line=dict(color='#1f77b4', width=3, shape='spline'),
            marker=dict(size=6, color='white', line=dict(width=2, color='#1f77b4')),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)',
            hovertemplate='<b>Year:</b> %{x}<br><b>PM2.5:</b> %{y:.2f} μg/m³<extra></extra>'
        ))
        
        # Plot forecasted upcoming trend
        forecast_future = forecast[forecast['ds'] > prophet_df['ds'].max()]
        fig_forecast.add_trace(go.Scatter(
            x=forecast_future['ds'].dt.year, 
            y=forecast_future['yhat'], 
            mode='lines+markers', 
            name='AI Forecast (10-Year Trend)',
            line=dict(color='#ff7f0e', width=3, dash='dot', shape='spline'),
            marker=dict(size=8, color='#ff7f0e', symbol='diamond'),
            hovertemplate='<b>Forecast Year:</b> %{x}<br><b>Expected PM2.5:</b> %{y:.2f} μg/m³<extra></extra>'
        ))
        
        # Add beautiful Confidence Intervals
        fig_forecast.add_trace(go.Scatter(
            x=pd.concat([forecast_future['ds'].dt.year, forecast_future['ds'].dt.year[::-1]]),
            y=pd.concat([forecast_future['yhat_upper'], forecast_future['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Forecast 80% Confidence Interval',
            hoverinfo='skip',
            showlegend=True
        ))
        
        # Premium Layout Tuning
        fig_forecast.update_layout(
            title=dict(text="Global PM2.5 Trajectory Forecast (Historical + AI Projection)", font=dict(size=20), x=0.5),
            xaxis_title="Year",
            yaxis_title="Average PM2.5 (μg/m³)",
            hovermode="x unified",
            height=450,
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0', gridwidth=1, dtick=5),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0', gridwidth=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not generate Prophet forecast. Error: {e}")

# ---------------------------------------------------
# REGIONAL ANALYSIS
# ---------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Compare Regional PM2.5 Trends")
    selected_regions = st.multiselect("Select Countries to Compare", sorted(df["Region"].dropna().unique()), default=["India", "China", "United States"])
    
    fig_region = go.Figure()
    for reg in selected_regions:
        reg_df = df[df["Region"] == reg].copy()
        fig_region.add_trace(go.Scatter(x=reg_df["year"], y=reg_df["pm25"], mode='lines+markers', name=f'{reg} Ground PM2.5'))
        # Plot the satellite proxy alongside the ground truth for each selected region
        if "MODIS_AOD_PROXY" in reg_df.columns and not reg_df["MODIS_AOD_PROXY"].isna().all():
            fig_region.add_trace(go.Scatter(x=reg_df["year"], y=reg_df["MODIS_AOD_PROXY"], mode='lines', name=f'{reg} Satellite Proxy', line=dict(dash='dot')))
    
    fig_region.update_layout(title="PM2.5 Trend Comparison", xaxis_title="Year", yaxis_title="PM2.5 (μg/m³)")
    st.plotly_chart(fig_region, use_container_width=True)
    
    # Automated Risk Alerts
    if selected_regions:
        st.markdown("**Automated AI Insights:**")
        for reg in selected_regions:
            latest = df[df["Region"] == reg].iloc[-1]
            level = "Good" if latest["pm25"] < 12 else "Moderate" if latest["pm25"] < 35 else "Unhealthy" if latest["pm25"] < 55 else "Hazardous"
            trend = "increase" if latest.get("pm25_growth", 0) > 0 else "decrease"
            growth_val = abs(latest.get("pm25_growth", 0) * 100)
            
            # Incorporate satellite reading if available
            sat_text = f" Satellite proxy reading is **{latest['MODIS_AOD_PROXY']:.1f}**." if pd.notna(latest.get("MODIS_AOD_PROXY")) else ""
            
            icon = "🟢" if level in ["Good", "Moderate"] else "⚠️"
            st.markdown(f"*{icon} **{reg}**: Current PM2.5 is **{latest['pm25']:.1f} μg/m³** ({level}). This is a **{growth_val:.1f}% {trend}** from the previous year.{sat_text}*")

with col2:
    st.subheader("🔮 Simulator: Predict Regional Pollution")
    st.markdown("Edit factors below to see what the machine learning model forecasts in real-time.")
    
    # Let user simulate for any selected region
    sim_region = st.selectbox("Select Country for Simulation", selected_regions if selected_regions else sorted(df["Region"].dropna().unique()))
    region_df = df[df["Region"] == sim_region].copy()
    
    col2a, col2b = st.columns(2)
    pop_input = col2a.number_input("Population (millions)", value=float(region_df.iloc[-1]['population']) if not region_df.empty else 10.0)
    sat_input = col2b.number_input("Satellite MODIS AOD Proxy", value=float(region_df.iloc[-1]['MODIS_AOD_PROXY']) if not region_df.empty and pd.notnull(region_df.iloc[-1]['MODIS_AOD_PROXY']) else 15.0)
    
    lag1_input = col2a.number_input("Previous Year PM2.5 (Lag1)", value=float(region_df.iloc[-1]['pm25']) if not region_df.empty else 20.0)
    lag2_input = col2b.number_input("2 Years Ago PM2.5 (Lag2)", value=float(region_df.iloc[-1].get('pm25_lag1', 20.0)) if not region_df.empty else 20.0)
    
    if st.button("Generate Forecast", use_container_width=True):
        if model is None:
            st.error("No model available.")
        elif not feature_names:
            st.error("Missing feature bindings for the model.")
        else:
            # Construct feature DF
            test_row = pd.DataFrame(columns=feature_names, index=[0])
            test_row.fillna(0, inplace=True) # Zero pad defaults
            
            # Smart-map known columns
            if "year" in feature_names: test_row["year"] = latest_year + 1
            if "log_population" in feature_names: test_row["log_population"] = np.log1p(pop_input)
            if "MODIS_AOD_PROXY" in feature_names: test_row["MODIS_AOD_PROXY"] = sat_input
            if "pm25_lag1" in feature_names: test_row["pm25_lag1"] = lag1_input
            if "pm25_lag2" in feature_names: test_row["pm25_lag2"] = lag2_input
            if "pm25_roll3_mean" in feature_names: test_row["pm25_roll3_mean"] = np.mean([lag1_input, lag2_input])
            
            predicted = model.predict(test_row)[0]
            st.success(f"**Predicted PM2.5: {predicted:.2f} µg/m³**")
            
            st.progress(min(int(predicted / 100 * 100), 100))
            if predicted < 12: st.info("Level: Good")
            elif predicted < 35: st.warning("Level: Moderate")
            else: st.error("Level: Unhealthy")

# ---------------------------------------------------
# FEATURE IMPORTANCE & SHAP EXPLAINABILITY
# ---------------------------------------------------
st.subheader("🧠 Model Transparency: What drives PM2.5?")

if model and feature_names:
    st.markdown("XGBoost decisions explained. Which global factors drive high pollution forecasts?")
    
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        st.markdown("**Global Feature Importance (XGBoost Weight)**")
        # Extract intrinsic feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            imp_df = pd.DataFrame({
                "Factor": feature_names,
                "Importance": importances
            }).sort_values("Importance", ascending=True)
            
            fig_imp = px.bar(imp_df, x="Importance", y="Factor", orientation='h',
                             title="Ranking of Influencing Factors",
                             color="Importance", color_continuous_scale="Reds")
            fig_imp.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Intrinsic feature importance not available for this model type.")
            
    with col_feat2:
        st.markdown("**SHAP Value Distribution (Directional Impact)**")
        valid_df = df.dropna(subset=feature_names)
        if not valid_df.empty:
            sample_size = min(500, len(valid_df))
            X_sample = valid_df[feature_names].sample(sample_size, random_state=42)
            
            try:
                plt.style.use('default')
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                fig_shap, ax_shap = plt.subplots(figsize=(6, 4))
                fig_shap.patch.set_facecolor('white')
                ax_shap.set_facecolor('white')
                
                shap.summary_plot(shap_values, X_sample, show=False)
                
                # Ensure text colors are black
                ax_shap.tick_params(colors='black')
                ax_shap.xaxis.label.set_color('black')
                ax_shap.yaxis.label.set_color('black')
                cb = fig_shap.axes[-1] if len(fig_shap.axes) > 1 else None
                if cb:
                    cb.tick_params(colors='black')
                    cb.yaxis.label.set_color('black')
                
                st.pyplot(fig_shap)
                
            except Exception as e:
                st.warning(f"Could not render SHAP plots. Error: {e}")
        else:
            st.info("Not enough valid data rows to compute SHAP.")

# ---------------------------------------------------
# DATA EXPLORER & DOWNLOAD
# ---------------------------------------------------
st.markdown("---")
st.subheader("🗄️ Raw Data Explorer & Export")
st.markdown("Filter, sort, and download the curated ground truth and satellite-derived data used in this dashboard.")

st.dataframe(df, use_container_width=True, height=300)

csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Full Dataset as CSV",
    data=csv,
    file_name='aerographfed_global_pm25.csv',
    mime='text/csv',
)
