import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats
import zipfile
import os
import glob
from pathlib import Path

# Paths
base_dir = Path('../CIESIN_SEDAC_SDEI_GWRPM25_MMSVAOD_5GL04_5.04-20260314_170047')
output_file = Path('../data/derived/satellite_pm25_features.csv')

def extract_satellite_features():
    import urllib.request
    import os
    
    # Get world map (Handling GeoPandas 1.0 deprecation of get_path)
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    zip_path = "ne_110m_admin_0_countries.zip"
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
    
    world = gpd.read_file(zip_path)
    
    # Natural Earth uses 'NAME' or 'ADMIN' depending on the version downloaded
    name_col = 'NAME' if 'NAME' in world.columns else 'ADMIN' if 'ADMIN' in world.columns else 'name'
    world['name'] = world[name_col].str.strip()
    
    results = []

    # Iterate through zip files
    zip_files = sorted(base_dir.glob('*geotiff.zip'))
    
    if not zip_files:
        # Fall back if we are running in the src dir directly without ../
        base_dir_fallback = Path('CIESIN_SEDAC_SDEI_GWRPM25_MMSVAOD_5GL04_5.04-20260314_170047')
        zip_files = sorted(base_dir_fallback.glob('*geotiff.zip'))

    for zf in zip_files:
        year_str = zf.stem.split('-')[-2] # e.g. from "...-2022-geotiff"
        if not year_str.isdigit():
            continue
        year = int(year_str)
        
        print(f"Processing year {year} from {zf.name}...")
        
        # Unzip temporarily
        temp_dir = Path(f'temp_{year}')
        temp_dir.mkdir(exist_ok=True)
        
        try:
            with zipfile.ZipFile(zf, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the .tif file
            tif_files = list(temp_dir.rglob('*.tif'))
            if not tif_files:
                print(f"No tif found in {zf.name}")
                continue
                
            tif_path = str(tif_files[0])
            
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                
                # A safer way to compute mean avoiding overflow is to use a custom function
                # but rasterstats 'mean' is usually sufficient if we ignore the warning. 
                # To be absolutely sure, we can pass a custom stat that casts to float64
                def safe_mean(x):
                    import numpy as np
                    return np.nanmean(x.astype(np.float64))
                    
                stats = zonal_stats(world, tif_path, stats=[], add_stats={'mean': safe_mean}, nodata=-9999)
            
            for i, stat in enumerate(stats):
                country_name = world.iloc[i]['name']
                # The 'mean' key holds the mean pixel value for the polygon
                mean_pm25 = stat['mean'] if stat['mean'] is not None else float('nan')
                
                results.append({
                    'Region': country_name,
                    'year': year,
                    'CIESIN_PM25': mean_pm25
                })
                
        finally:
            # Cleanup temp files
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(temp_dir)
            
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Feature extraction complete! Saved to {output_file}")


if __name__ == "__main__":
    extract_satellite_features()
