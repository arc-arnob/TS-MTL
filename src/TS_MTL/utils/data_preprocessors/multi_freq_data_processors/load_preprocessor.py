import pandas as pd
import os
import numpy as np

def process_load_forecasting_data(output_dir, load_file, temp_file):
    """
    Process load forecasting data and save to separate HF and LF files.
    Handles common data format issues.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the load history file - handle European number format with commas
    load_df = pd.read_csv(load_file, thousands=',')
    
    # Read the temperature history file - handle European number format with commas
    temp_df = pd.read_csv(temp_file, thousands=',')
    
    print(f"Load data shape: {load_df.shape}")
    print(f"Temperature data shape: {temp_df.shape}")
    
    # Make sure numeric columns are properly converted
    for col in load_df.columns:
        if col not in ['zone_id', 'year', 'month', 'day']:
            load_df[col] = pd.to_numeric(load_df[col], errors='coerce')
    
    for col in temp_df.columns:
        if col not in ['station_id', 'year', 'month', 'day']:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
    
    # Fill any NaN values created during conversion using modern pandas syntax
    load_df = load_df.ffill().bfill()
    temp_df = temp_df.ffill().bfill()
    
    # Process each zone (1-20)
    for zone_id in range(1, 21):
        print(f"Processing zone {zone_id}...")
        
        # Filter load data for this zone
        zone_load = load_df[load_df['zone_id'] == zone_id].copy()
        
        # Skip if no data for this zone
        if len(zone_load) == 0:
            print(f"  No data for zone {zone_id}, skipping")
            continue
        
        # For simplicity, let's associate each zone with the geographically closest weather stations
        # Without actual geographic data, we'll use a simple approach based on zone_id
        # You should replace this with a more accurate mapping if available
        
        # For each zone, use 3 stations: one matching the zone_id (modulo 11), 
        # plus the stations before and after it
        primary_station = ((zone_id - 1) % 11) + 1
        station_ids = [
            primary_station,
            ((primary_station) % 11) + 1,  # next station (wraps around)
            ((primary_station - 2) % 11) + 1  # previous station (wraps around)
        ]
        
        # Create processed dataframes for this zone
        zone_lf = pd.DataFrame()
        zone_hf = pd.DataFrame()
        
        # Convert wide format (hourly columns) to long format for load data
        for idx, row in zone_load.iterrows():
            year = int(row['year'])
            month = int(row['month'])
            day = int(row['day'])
            
            for hour in range(1, 25):
                hour_col = f'h{hour}'
                if hour_col in row:
                    # Create timestamp
                    timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour-1)
                    
                    # Add to zone_lf dataframe
                    zone_lf = pd.concat([zone_lf, pd.DataFrame({
                        'Time': [timestamp],
                        'load': [float(row[hour_col])]
                    })])
        
        # Create high-frequency data from temperature stations
        for station_id in station_ids:
            # Filter temperature data for this station
            station_temp = temp_df[temp_df['station_id'] == station_id].copy()
            
            # Skip if no data for this station
            if len(station_temp) == 0:
                print(f"  No data for station {station_id}, skipping")
                continue
                
            # Convert wide format to long format for temperature data
            station_data = pd.DataFrame()
            
            for idx, row in station_temp.iterrows():
                year = int(row['year'])
                month = int(row['month'])
                day = int(row['day'])
                
                for hour in range(1, 25):
                    hour_col = f'h{hour}'
                    if hour_col in row:
                        # Create timestamp
                        timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour-1)
                        
                        # Add to station_data dataframe
                        station_data = pd.concat([station_data, pd.DataFrame({
                            'Time': [timestamp],
                            f'temp_station_{station_id}': [float(row[hour_col])]
                        })])
            
            # If this is the first station, use it as the base for zone_hf
            if len(zone_hf) == 0:
                zone_hf = station_data.copy()
            else:
                # Otherwise merge with existing zone_hf
                zone_hf = pd.merge(
                    zone_hf, 
                    station_data,
                    on='Time', 
                    how='outer'
                )
        
        # Make sure dataframes are sorted by time
        zone_lf = zone_lf.sort_values('Time')
        zone_hf = zone_hf.sort_values('Time')
        
        # Fill any missing values using modern pandas syntax
        zone_hf = zone_hf.ffill().bfill()
        
        # Create 15-minute HF data by interpolation
        zone_hf.set_index('Time', inplace=True)
        hf_resampled = zone_hf.resample('15min').interpolate(method='linear')
        hf_resampled.reset_index(inplace=True)
        
        # Add time-based features
        hf_resampled['hour_sin'] = np.sin(2 * np.pi * hf_resampled['Time'].dt.hour / 24)
        hf_resampled['hour_cos'] = np.cos(2 * np.pi * hf_resampled['Time'].dt.hour / 24)
        hf_resampled['day_of_week'] = hf_resampled['Time'].dt.dayofweek
        hf_resampled['month_sin'] = np.sin(2 * np.pi * hf_resampled['Time'].dt.month / 12)
        hf_resampled['month_cos'] = np.cos(2 * np.pi * hf_resampled['Time'].dt.month / 12)
        
        # Save files
        lf_filename = f"{output_dir}/zone-{zone_id}-lf.csv"
        hf_filename = f"{output_dir}/zone-{zone_id}-hf.csv"
        
        zone_lf.to_csv(lf_filename, index=False)
        hf_resampled.to_csv(hf_filename, index=False)
        
        print(f"Saved {lf_filename} and {hf_filename}")
        print(f"  LF shape: {zone_lf.shape}")
        print(f"  HF shape: {hf_resampled.shape}")
        print(f"  Frequency ratio: 4:1 (15-minute temp data : hourly load data)")

def main():
    # Define paths with robust path handling
    output_dir = "./src/TS_MTL/data/load"
    load_file = "./raw_data/energy/Load_history.csv"
    temp_file = "./raw_data/energy/Temperature_history.csv"
    
    # Process and save data
    process_load_forecasting_data(output_dir, load_file, temp_file)
    
    print("Data processing complete!")

if __name__ == "__main__":
    main()