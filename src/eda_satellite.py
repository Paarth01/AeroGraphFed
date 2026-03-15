import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Original Data
try:
    df_global = pd.read_csv('../data/raw/global_data.csv')
except FileNotFoundError:
    df_global = pd.read_csv('data/raw/global_data.csv')

# 2. Load New Satellite Data
try:
    df_sat = pd.read_csv('../data/derived/satellite_pm25_features.csv')
except FileNotFoundError:
    df_sat = pd.read_csv('data/derived/satellite_pm25_features.csv')

# 3. Standardize Columns before merging
if 'Year' in df_global.columns:
    df_global = df_global.rename(columns={'Year': 'year'})

print(f"Original Records: {len(df_global)}")
print(f"Satellite Records: {len(df_sat)}")

# 4. Merge
df_merged = pd.merge(df_global, df_sat, on=['Region', 'year'], how='inner')
print(f"Merged Records (Inner Join): {len(df_merged)}")

if len(df_merged) == 0:
    print("WARNING: No rows matched. Likely country name mismatch between datasets.")
    print("Sample Global Regions:", df_global['Region'].unique()[:5])
    print("Sample Satellite Regions:", df_sat['Region'].unique()[:5])
else:
    # 5. Correlation Analysis
    target_col = 'Population-Weighted PM2.5 [ug/m3]'
    if target_col in df_merged.columns:
        corr = df_merged[['CIESIN_PM25', target_col]].corr().iloc[0, 1]
        print(f"\nCorrelation between Extracted CIESIN GeoTIFF PM2.5 and Original Ground PM2.5: {corr:.4f}")
        
        # 6. Scatter Plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_merged, x='CIESIN_PM25', y=target_col, alpha=0.5)
        
        # Add perfect correlation line
        max_val = max(df_merged['CIESIN_PM25'].max(), df_merged[target_col].max())
        plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect 1:1 Match (r=1.0)')
        
        plt.title('Extracted CIESIN Satellite PM2.5 vs Original Target PM2.5')
        plt.xlabel('Satellite GeoTIFF PM2.5 (μg/m³)')
        plt.ylabel('Original Population-Weighted PM2.5 (μg/m³)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('ciesin_correlation_plot.png')
        print("\nEDA Plot saved to 'ciesin_correlation_plot.png'")
        
        # 7. Distribution Comparison
        plt.figure(figsize=(10, 5))
        sns.kdeplot(df_merged['CIESIN_PM25'], label='Extracted CIESIN', fill=True)
        sns.kdeplot(df_merged[target_col], label='Original Pop-Weighted', fill=True)
        plt.title('PM2.5 Distribution Comparison')
        plt.xlabel('PM2.5 (μg/m³)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('ciesin_distribution_plot.png')
        print("Distribution Plot saved to 'ciesin_distribution_plot.png'")
