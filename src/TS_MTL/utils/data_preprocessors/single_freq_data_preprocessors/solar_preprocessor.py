# import os
# import pandas as pd

# def process_actual_files_to_columns(input_dir, output_file):
#     """
#     Process files in a directory:
#     - Deletes non-`Actual_` files.
#     - Combines all `Actual_` files into a single CSV, creating one column per file.
#     - Keeps the shared timestamp column and renames file-specific columns sequentially as `loc-x`.

#     Args:
#         input_dir (str): Directory containing the files.
#         output_file (str): Path to save the combined CSV file.
#     """
#     actual_files = []
    
#     # Iterate through files in the directory
#     for filename in os.listdir(input_dir):
#         if filename.startswith("Actual_") and filename.endswith(".csv"):
#             # Collect Actual files
#             actual_files.append(os.path.join(input_dir, filename))
#         else:
#             # Delete non-Actual files
#             os.remove(os.path.join(input_dir, filename))
#             print(f"Deleted: {filename}")

#     # Initialize an empty DataFrame for combining
#     combined_df = None
#     max_locations = 10
#     # Limit to max_locations
#     actual_files = actual_files[:max_locations]

#     # Process each Actual file
#     for i, file_path in enumerate(actual_files, start=1):
#         # Read the file
#         df = pd.read_csv(file_path)

#         # Rename the last column of the current file to `loc-x`
#         df.rename(columns={df.columns[-1]: f"loc-{i}"}, inplace=True)

#         # Merge with the combined DataFrame based on the timestamp
#         if combined_df is None:
#             combined_df = df
#         else:
#             combined_df = pd.merge(combined_df, df[['LocalTime', f"loc-{i}"]], on='LocalTime', how='inner')
        
#         print(f"Processed: {os.path.basename(file_path)}")

#     # Rename the last column in the combined DataFrame to 'OT'
#     combined_df.rename(columns={combined_df.columns[-1]: "OT"}, inplace=True)
#     # rename LocalTime to date
#     combined_df.rename(columns={"LocalTime": "date"}, inplace=True)
#     # Convert 'date' to datetime format
#     combined_df['date'] = pd.to_datetime(combined_df['date'])
#     # Save the combined DataFrame to a CSV file
#     combined_df.to_csv(output_file, index=False)
#     print(f"Combined file saved to: {output_file}")


# input_directory = "../dataset/zips/al-pv-2006"
# output_csv = "solar_al.csv"

# process_actual_files_to_columns(input_directory, output_csv)

# input_directory = "../dataset/zips/fl-pv-2006"
# output_csv = "solar_fl.csv"
# process_actual_files_to_columns(input_directory, output_csv)

# input_directory = "../dataset/zips/il-pv-2006"
# output_csv = "solar_il.csv"

# process_actual_files_to_columns(input_directory, output_csv)

# input_directory = "../dataset/zips/ks-pv-2006"
# output_csv = "solar_ks.csv"

# process_actual_files_to_columns(input_directory, output_csv)

# input_directory = "../dataset/zips/ma-pv-2006"
# output_csv = "solar_ma.csv"
# process_actual_files_to_columns(input_directory, output_csv)
# input_directory = "../dataset/zips/me-pv-2006"
# output_csv = "solar_me.csv"
# process_actual_files_to_columns(input_directory, output_csv)


import os
import pandas as pd
import dateutil.parser

def filter_and_save_solar_data(input_dir, min_date=None, max_date=None):
    """
    Loads all CSVs from a directory, filters them based on min_date and max_date, 
    and saves them back to the same location.

    Args:
    - input_dir (str): Path to the directory containing CSV files.
    - min_date (str or datetime, optional): Start date for filtering.
    - max_date (str or datetime, optional): End date for filtering.
    """
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for csv_file in csv_files:
        file_path = os.path.join(input_dir, csv_file)
        
        # Load CSV
        df = pd.read_csv(file_path)

        # Ensure 'date' column is in datetime format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

            # Apply date filtering if provided
            if min_date:
                min_date = dateutil.parser.parse(min_date) if isinstance(min_date, str) else min_date
                df = df[df['date'] >= min_date]

            if max_date:
                max_date = dateutil.parser.parse(max_date) if isinstance(max_date, str) else max_date
                df = df[df['date'] <= max_date]

            # Save the filtered data back
            df.to_csv(file_path, index=False)
            print(f"Filtered and saved: {file_path}")

# Define locations and apply filtering
solar_dirs = ["../dataset/solar_al", "../dataset/solar_fl", "../dataset/solar_il",
              "../dataset/solar_ks", "../dataset/solar_ma", "../dataset/solar_me"]

min_date = "2006-09-01"
max_date = "2006-09-08 04:50"

for solar_dir in solar_dirs:
    if os.path.exists(solar_dir):
        filter_and_save_solar_data(solar_dir, min_date, max_date)
    else:
        print(f"Directory not found: {solar_dir}")



