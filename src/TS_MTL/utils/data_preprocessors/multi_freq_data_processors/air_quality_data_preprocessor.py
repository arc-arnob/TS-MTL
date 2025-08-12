import pandas as pd
import numpy as np
import os
from datetime import datetime

def process_airquality_data(output_dir, station_files):
    """
    Process air quality data and save to separate HF and LF files.
    Target variable CO is in LF, all other variables are in HF.
    Output files will use station numbers instead of names.
    
    Args:
        output_dir: Directory to save output files
        station_files: Dictionary mapping station names to file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define which columns go in which frequency
    # CO is our target, so it goes in LF
    lf_columns = ['CO']  # Target variable as LF
    # All other variables go in HF
    hf_columns = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3']
    
    # Convert dictionary to list to ensure consistent ordering
    station_items = list(station_files.items())
    
    # Process each station with consistent numbering
    for i in range(len(station_items)):
        station_name, file_path = station_items[i]
        station_id = i + 1  # Use consistent sequential numbers starting from 1
        
        print(f"Processing station {station_id}/{len(station_items)}: {station_name}")
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            print(f"  Loaded {len(df)} records")
            
            # Create date column and drop unnecessary columns
            df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
            df.drop(columns=['No', 'year', 'month', 'day', 'hour'], inplace=True, errors='ignore')
            
            # Handle missing values using modern pandas syntax
            df = df.ffill()
            df = df.bfill()
            df = df.interpolate(method='linear', limit_direction='both')
            
            # Convert columns to numeric (if needed)
            for col in df.columns:
                if col not in ['date', 'station', 'wd']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create separate HF and LF dataframes
            # Make sure all required columns exist in the dataframe
            available_hf_cols = [col for col in hf_columns if col in df.columns]
            available_lf_cols = [col for col in lf_columns if col in df.columns]
            
            if not available_lf_cols:
                print(f"  Warning: Missing CO column for station {station_name}, skipping")
                continue
            
            df_hf = df[['date'] + available_hf_cols].copy()
            df_lf = df[['date'] + available_lf_cols].copy()
            
            # Set date as index
            df_hf.set_index('date', inplace=True)
            df_lf.set_index('date', inplace=True)
            
            # Resample to create the multi-frequency structure
            # LF data stays at hourly frequency
            # Resample HF data to 15-minute frequency (4:1 ratio)
            df_hf_resampled = df_hf.resample('15min').interpolate(method='cubic')
            
            # Add time features to HF data
            df_hf_resampled['hour_sin'] = np.sin(2 * np.pi * df_hf_resampled.index.hour / 24)
            df_hf_resampled['hour_cos'] = np.cos(2 * np.pi * df_hf_resampled.index.hour / 24)
            df_hf_resampled['day_sin'] = np.sin(2 * np.pi * df_hf_resampled.index.day / 31)
            df_hf_resampled['day_cos'] = np.cos(2 * np.pi * df_hf_resampled.index.day / 31)
            df_hf_resampled['month_sin'] = np.sin(2 * np.pi * df_hf_resampled.index.month / 12)
            df_hf_resampled['month_cos'] = np.cos(2 * np.pi * df_hf_resampled.index.month / 12)
            df_hf_resampled['weekday'] = df_hf_resampled.index.weekday
            
            # Reset index to have date as a column
            df_hf_resampled.reset_index(inplace=True)
            df_hf_resampled.rename(columns={'index': 'Time', 'date': 'Time'}, inplace=True)
            
            df_lf.reset_index(inplace=True)
            df_lf.rename(columns={'index': 'Time', 'date': 'Time'}, inplace=True)
            
            # Use station_id (numeric) for filenames - force consistent numbering
            hf_filename = f"{output_dir}/station-{station_id}-hf.csv"
            lf_filename = f"{output_dir}/station-{station_id}-lf.csv"
            
            # Print debug info
            print(f"  Creating files for station ID: {station_id}")
            
            df_hf_resampled.to_csv(hf_filename, index=False)
            df_lf.to_csv(lf_filename, index=False)
            
            print(f"  Saved files: {hf_filename}, {lf_filename}")
            print(f"  HF shape: {df_hf_resampled.shape}, LF shape: {df_lf.shape}")
            print(f"  Frequency ratio: 4:1 (15-minute data : hourly CO data)")
            
        except Exception as e:
            print(f"  Error processing {station_name}: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    # Define station files with robust path handling
    # These paths are relative to the project root directory
    station_files = {
        "1": "./raw_data/air_quality/PRSA_Data_Aotizhongxin.csv",
        "2": "./raw_data/air_quality/PRSA_Data_Dingling.csv",
        "3": "./raw_data/air_quality/PRSA_Data_Gucheng.csv",
        "4": "./raw_data/air_quality/PRSA_Data_Huairou.csv",
        "5": "./raw_data/air_quality/PRSA_Data_Tiantan.csv",
        "6": "./raw_data/air_quality/PRSA_Data_Wanshouxigong.csv"
    }
    
    # Define output directory
    output_dir = "./src/TS_MTL/data/air_quality"
    
    # Process and save data
    process_airquality_data(output_dir, station_files)
    
    print("Air quality data processing complete!")

if __name__ == "__main__":
    main()