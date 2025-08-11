import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

def process_spain_multisite_data(input_file, output_dir):
    """
    Process Spain multi-site load data and save individual city files.
    
    Args:
        input_file: Path to the main Spain dataset file
        output_dir: Directory to save individual city files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the data
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Extract city-specific features
    city_cols = [col for col in df.columns if '_' in col and col != 'load_shortfall_3h']
    cities = sorted(set(col.split('_')[0] for col in city_cols))
    print(f"Found {len(cities)} cities: {cities}")
    
    # Build a mapping: city → [feature_name]
    city_to_feats = defaultdict(set)
    for col in city_cols:
        city, feat = col.split('_', 1)
        city_to_feats[city].add(feat)
    
    # Find common features across all cities
    common_feats = set.intersection(*city_to_feats.values())
    # Drop unwanted feature(s)
    common_feats.discard("pressure")
    print(f"Common features across all cities: {sorted(common_feats)}")
    
    # Save CSV per city with common features and no city prefix
    saved_files = []
    for city in cities:
        print(f"Processing city: {city}")
        
        prefixed_cols = [f"{city}_{feat}" for feat in common_feats if f"{city}_{feat}" in df.columns]
        city_df = df[["time"] + prefixed_cols + ["load_shortfall_3h"]]
        
        # Rename: remove city prefix from columns
        rename_map = {f"{city}_{feat}": feat for feat in common_feats}
        city_df = city_df.rename(columns=rename_map)
        
        # Save
        output_file = f"{output_dir}/{city}.csv"
        city_df.to_csv(output_file, index=False)
        saved_files.append(output_file)
        print(f"  Saved: {output_file} (shape: {city_df.shape})")
    
    print(f"✅ Saved {len(cities)} city files with {len(common_feats)} common features each, without city prefixes.")
    return saved_files

def main():
    """
    Main function for Spain multi-site load data preprocessing.
    Uses project-relative paths that work from any working directory.
    """
    # Define paths with robust path handling
    input_file = "./raw_data/spain/df_train.csv"
    output_dir = "./src/TS_MTL/data/spain_mf"
    
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    # Process and save data
    saved_files = process_spain_multisite_data(input_file, output_dir)
    
    print("Spain multi-site load data processing complete!")
    print(f"Created {len(saved_files)} city files in: {output_dir}")

if __name__ == "__main__":
    main()
