import pandas as pd
import os
import dateutil.parser

class AirQualityProcessor:
    def __init__(self, data, min_date=None, max_date=None):
        self.data = data
        self.forecast_columns = None
        self.min_date = min_date
        self.max_date = max_date

    def preprocess_airquality_data(self):
        """Preprocess the Air Quality dataset."""
        # Create 'date' column by combining 'year', 'month', 'day', and 'hour' if they exist
        if {'year', 'month', 'day', 'hour'}.issubset(self.data.columns):
            self.data['date'] = pd.to_datetime(self.data[['year', 'month', 'day', 'hour']])
            # Drop the original columns after creating 'date'
            self.data.drop(columns=['year', 'month', 'day', 'hour'], inplace=True)
        else:
            # Ensure 'date' column exists as datetime
            self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Filter data based on the provided min_date and max_date
        if self.min_date:
            min_date_parsed = dateutil.parser.parse(self.min_date) if isinstance(self.min_date, str) else self.min_date
            self.data = self.data[self.data['date'] >= min_date_parsed]

        if self.max_date:
            max_date_parsed = dateutil.parser.parse(self.max_date) if isinstance(self.max_date, str) else self.max_date
            self.data = self.data[self.data['date'] <= max_date_parsed]

        # Set 'date' as the index
        self.data.set_index('date', inplace=True)

        # Drop unnecessary columns
        columns_to_drop = ['No', 'wd', 'station']
        self.data.drop(columns=[col for col in columns_to_drop if col in self.data.columns], inplace=True)

        # Rename the 'CO' column to 'OT' if it exists
        if 'CO' in self.data.columns:
            self.data.rename(columns={'CO': 'OT'}, inplace=True)

        # Fill missing values
        self.data.fillna(method='ffill', inplace=True)  # Forward fill
        self.data.fillna(method='bfill', inplace=True)  # Backward fill
        self.data.interpolate(method='linear', inplace=True, limit_direction='both')  # Interpolate missing values

        # Move 'OT' column to the end
        if 'OT' in self.data.columns:
            columns = [col for col in self.data.columns if col != 'OT'] + ['OT']
            self.data = self.data[columns]

        # Reset the index
        self.data.reset_index(inplace=True)
        print(self.data.head())

# Driver Code
def process_and_save_sites():
    for i in range(1, 13):  # Loop through site-1 to site-12
        input_file = f"../dataset/air_quality_cluster/site-{i}.csv"  # Input file name
        output_dir = f"../dataset/site-{i}"      # Output directory
        output_file = os.path.join(output_dir, f"site-{i}.csv")  # Full path for the output file

        # Check if the input file exists
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist. Skipping...")
            continue

        # Load the data
        print(f"Processing {input_file}...")
        data = pd.read_csv(input_file)

        # Process the data
        processor = AirQualityProcessor(data, min_date='2014-09-01', max_date='2014-11-12 19:00')
        processor.preprocess_airquality_data()

        # Save the processed file
        processor.data.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")

# Call the driver function
process_and_save_sites()
