import os
import pandas as pd

def process_store_data(df, output_dir="stores_data"):
    """
    Process the dataframe:
    - For each Store, create a new CSV file: store-x/store-x.csv.
    - Rename the 'Sales' column to 'OT'.
    - Merge 'StateHoliday' and 'SchoolHoliday' columns into a single column 'Holiday'.
    - Handle non-numeric values in 'StateHoliday' and 'SchoolHoliday'.
    - Change 'date' to datetime format.

    Args:
        df (pd.DataFrame): Input dataframe with store data.
        output_dir (str): Root directory to save store-specific files.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each unique store ID
    for store_id in df['Store'].unique():
        # Filter data for the current store
        store_df = df[df['Store'] == store_id].copy()

        # Rename the 'Sales' column to 'OT'
        store_df.rename(columns={'Sales': 'OT'}, inplace=True)

        # Handle non-numeric values in 'StateHoliday' and 'SchoolHoliday'
        for col in ['StateHoliday', 'SchoolHoliday']:
            store_df[col] = pd.to_numeric(store_df[col], errors='coerce').fillna(0).astype(int)

        # Combine 'StateHoliday' and 'SchoolHoliday' into a single 'Holiday' column
        store_df['Holiday'] = store_df['StateHoliday'] + store_df['SchoolHoliday']

        # Drop the original 'StateHoliday' and 'SchoolHoliday' columns
        store_df.drop(columns=['StateHoliday', 'SchoolHoliday'], inplace=True)

        # Convert all column names to lowercase except 'OT'
        store_df.columns = [col.lower() if col != 'OT' else col for col in store_df.columns]

        # Change 'date' to datetime format
        store_df['date'] = pd.to_datetime(store_df['date'])

        # Create a subdirectory for the store
        store_dir = os.path.join(output_dir, f"store-{store_id}")
        os.makedirs(store_dir, exist_ok=True)

        # Save the processed data to a CSV file
        output_file = os.path.join(store_dir, f"store-{store_id}.csv")
        store_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")


# Create a pandas dataframe
df = pd.read_csv('../dataset/sales/train.csv')

# Process and save the store data
process_store_data(df)
