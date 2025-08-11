#!/bin/bash

# Wind Power Data Preprocessor Runner
# This script processes raw wind power and forecast data

set -e  # Exit on any error

echo "=== Wind Power Data Preprocessor ==="
echo "Converting raw wind power and forecast data to mixed-frequency format..."

# Check if we're in the right directory
if [ ! -d "src/TS_MTL" ]; then
    echo "Error: Please run this script from the TS-MTL root directory"
    exit 1
fi

# Create necessary directories
echo "Creating output directories..."
mkdir -p ./src/TS_MTL/data/wind
mkdir -p ./raw_data/wind

# Check if raw data exists
RAW_DIR="./raw_data/wind"
EXPECTED_FILES=(
    "train.csv"
    "windforecasts_wf1.csv"
    "windforecasts_wf2.csv"
    "windforecasts_wf3.csv"
    "windforecasts_wf4.csv"
    "windforecasts_wf5.csv"
    "windforecasts_wf6.csv"
    "windforecasts_wf7.csv"
)

echo "Checking for raw data files..."
missing_files=0
for file in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "$RAW_DIR/$file" ]; then
        echo "  Missing: $RAW_DIR/$file"
        missing_files=$((missing_files + 1))
    else
        echo "  Found: $RAW_DIR/$file"
    fi
done

if [ $missing_files -gt 0 ]; then
    echo ""
    echo "Error: Missing $missing_files raw data files."
    echo "Please download the wind power forecasting dataset and place the CSV files in:"
    echo "  $RAW_DIR/"
    echo ""
    echo "Expected files:"
    for file in "${EXPECTED_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Dataset source: Wind power forecasting competition data (e.g., GEFCom2012 Wind Track)"
    exit 1
fi

# Set PYTHONPATH if TS_MTL package is not installed
if ! python -c "import TS_MTL" 2>/dev/null; then
    echo "Setting PYTHONPATH..."
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
fi

# Run the preprocessor
echo "Running wind power data preprocessor..."
python -c "
import sys
sys.path.append('src')
from TS_MTL.utils.data_preprocessors.wind_preprocessor import main
main()
"

# Check if output files were created
OUTPUT_DIR="./src/TS_MTL/data/wind"
echo "Checking output files..."
for i in {1..7}; do
    if [ -f "$OUTPUT_DIR/wind-farm-$i-hf.csv" ] && [ -f "$OUTPUT_DIR/wind-farm-$i-lf.csv" ]; then
        echo "  ✓ Wind farm $i files created"
    else
        echo "  ✗ Wind farm $i files missing"
    fi
done

echo ""
echo "Wind power data preprocessing complete!"
echo "Output files saved to: $OUTPUT_DIR"
echo "You can now run wind power experiments by updating your config to use:"
echo "  data.base_path: 'src/TS_MTL/data/wind'"
echo "  data.sites: [wind-farm-1, wind-farm-2, ..., wind-farm-7]"
