import os
import pandas as pd

def save_crypto_files_for_selected_assets(df, output_dir="crypto-data", selected_assets=[0, 1, 2, 3, 4, 5]):
    """
    Process the crypto dataset:
    - Filter for specific Asset_IDs (0, 1, 2, 3, 4, 5).
    - Resample data into hourly frequency with correct aggregation functions.
    - Drop specified columns ('timestamp', 'Target', 'Count', 'VWAP') and save results.
    - Round all column values to one decimal place.

    Args:
        df (pd.DataFrame): Input dataframe containing crypto data.
        output_dir (str): Directory to save the files. Defaults to "crypto-data".
        selected_assets (list): List of Asset_IDs to process. Defaults to [0, 1, 2, 3, 4, 5].
    """
    # Ensure the output directory exists
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert 'timestamp' to datetime
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')

    # Select data within a six-month period
    start_date = df['date'].min()
    end_date = start_date + pd.DateOffset(months=6)
    filtered_df = df[(df['date'] >= start_date) & (df['date'] < end_date)]

    # Filter for selected Asset_IDs
    filtered_df = filtered_df[filtered_df['Asset_ID'].isin(selected_assets)]

    # Define aggregation dictionary for resampling
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }

    # Get all unique Asset_IDs
    for asset_id in filtered_df['Asset_ID'].unique():
        # Filter data for the current Asset_ID
        asset_df = filtered_df[filtered_df['Asset_ID'] == asset_id].copy()

        # Set index to 'timestamp' and resample using the aggregation dictionary
        asset_df = asset_df.set_index('timestamp').resample('H').agg(agg_dict)

        # Reindex and interpolate missing values
        all_dates = pd.date_range(start=asset_df.index.min(), end=asset_df.index.max(), freq='H')
        asset_df = asset_df.reindex(all_dates).interpolate().reset_index().fillna(0)

        # Rename the index and columns
        asset_df.rename(columns={'index': 'timestamp', 'Close': 'OT'}, inplace=True)

        # Round all values to one decimal place
        asset_df = asset_df.round(1)

        # Drop unnecessary columns
        asset_df.drop(columns=['Target', 'Count', 'VWAP'], inplace=True, errors='ignore')

        # Create a subdirectory for the current asset
        asset_dir = os.path.join(output_dir, f"crypto-{asset_id}")
        os.makedirs(asset_dir, exist_ok=True)

        # Save the processed data to a CSV file
        output_file = os.path.join(asset_dir, f"crypto-{asset_id}.csv")
        asset_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

# Load the dataset and run the function
crypto_df = pd.read_csv("../dataset/crypto/crypto.csv")
save_crypto_files_for_selected_assets(crypto_df)
