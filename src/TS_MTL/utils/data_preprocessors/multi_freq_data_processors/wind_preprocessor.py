import pandas as pd
import os
import numpy as np
from datetime import datetime

def process_wind_farm_data(output_dir, wind_power_file, wind_forecast_files):
    """
    Process wind farm data and save to separate HF and LF files.
    
    Args:
        output_dir: Directory to save output files
        wind_power_file: Path to the main wind power file (LF data)
        wind_forecast_files: List of paths to wind forecast files (HF data)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the wind power file (LF data)
    wp_df = pd.read_csv(wind_power_file)
    
    # Convert date column to datetime
    wp_df['date'] = pd.to_datetime(wp_df['date'], format='%Y%m%d%H')
    wp_df = wp_df.rename(columns={'date': 'Time'})
    
    # Process each wind farm
    for site_id, forecast_file in enumerate(wind_forecast_files, 1):
        print(f"Processing wind farm {site_id}...")
        
        # Get wind power data for this site
        site_col = f'wp{site_id}'
        lf_data = wp_df[['Time', site_col]].copy()
        
        # Rename the wind power column to 'wp' for consistency across all sites
        lf_data = lf_data.rename(columns={site_col: 'wp'})
        
        # Read forecast data (HF data)
        wf_df = pd.read_csv(forecast_file)
        wf_df['date'] = pd.to_datetime(wf_df['date'], format='%Y%m%d%H')
        
        # Create a new datetime column for the actual forecast time
        wf_df['Time'] = wf_df.apply(
            lambda row: row['date'] + pd.Timedelta(hours=int(row['hors'])), 
            axis=1
        )
        
        # Keep only relevant columns
        hf_data = wf_df[['Time', 'u', 'v', 'ws', 'wd']].copy()
        
        # Sort by time
        lf_data = lf_data.sort_values('Time')
        hf_data = hf_data.sort_values('Time')
        
        # Fill missing values using modern pandas syntax
        lf_data = lf_data.ffill()
        hf_data = hf_data.ffill()
        
        # Resample LF data to daily frequency
        lf_data_resampled = lf_data.resample('1D', on="Time").median().reset_index()
        
        # Resample HF data to 6-hour frequency (4 times per day)
        hf_data_resampled = hf_data.resample('6H', on="Time").median().reset_index()
        
        # Interpolate any remaining missing values
        lf_data_resampled = lf_data_resampled.interpolate(method="linear")
        hf_data_resampled = hf_data_resampled.interpolate(method="linear")
        
        # Save files
        lf_filename = f"{output_dir}/wind-farm-{site_id}-lf.csv"
        hf_filename = f"{output_dir}/wind-farm-{site_id}-hf.csv"
        
        lf_data_resampled.to_csv(lf_filename, index=False)
        hf_data_resampled.to_csv(hf_filename, index=False)
        
        print(f"Saved {lf_filename} and {hf_filename}")
        print(f"  LF shape: {lf_data_resampled.shape}")
        print(f"  HF shape: {hf_data_resampled.shape}")

def main():
    # Define paths with robust path handling
    output_dir = "./src/TS_MTL/data/wind"
    wind_power_file = "./raw_data/wind/train.csv"  # Path to the main wind power file
    
    # Create paths to all 7 wind forecast files
    wind_forecast_files = [f"./raw_data/wind/windforecasts_wf{i}.csv" for i in range(1, 8)]
    
    # Process and save data
    process_wind_farm_data(output_dir, wind_power_file, wind_forecast_files)
    
    print("Data processing complete!")

if __name__ == "__main__":
    main()