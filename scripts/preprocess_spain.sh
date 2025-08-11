#!/bin/bash

# Spain Load Data Preprocessor Runner
# This script processes raw Spain multi-site load data

set -e  # Exit on any error

echo "=== Spain Load Data Preprocessor ==="
echo "Converting raw Spain multi-site load data to mixed-frequency format..."

# Check if we're in the right directory
if [ ! -d "src/TS_MTL" ]; then
    echo "Error: Please run this script from the TS-MTL root directory"
    exit 1
fi

# Create necessary directories
echo "Creating output directories..."
mkdir -p ./src/TS_MTL/data/spain_mf
mkdir -p ./raw_data/spain_multi_site

# Check if raw data exists
RAW_DIR="./raw_data/spain_multi_site"
EXPECTED_FILES=(
    "df_train.csv"
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
    echo "Please download the Spain multi-site load dataset and place the CSV files in:"
    echo "  $RAW_DIR/"
    echo ""
    echo "Expected files:"
    for file in "${EXPECTED_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Dataset source: Spain electricity load forecasting data"
    exit 1
fi

# Set PYTHONPATH if TS_MTL package is not installed
if ! python -c "import TS_MTL" 2>/dev/null; then
    echo "Setting PYTHONPATH..."
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
fi

# Run the preprocessor
echo "Running Spain load data preprocessor..."
cd src/TS_MTL/utils/data_preprocessors
python spain_data_preprocessor.py
cd ../../../../

# Move the output files to the correct location
if [ -d "src/TS_MTL/utils/data_preprocessors/cities_common" ]; then
    echo "Moving processed files to output directory..."
    mv src/TS_MTL/utils/data_preprocessors/cities_common/* ./src/TS_MTL/data/spain_mf/
    rmdir src/TS_MTL/utils/data_preprocessors/cities_common
fi

# Check if output files were created
OUTPUT_DIR="./src/TS_MTL/data/spain_mf"
echo "Checking output files..."
file_count=$(ls -1 "$OUTPUT_DIR"/*.csv 2>/dev/null | wc -l)
if [ $file_count -gt 0 ]; then
    echo "  ✓ Created $file_count city files"
    ls "$OUTPUT_DIR"/*.csv | head -5
    if [ $file_count -gt 5 ]; then
        echo "  ... and $((file_count - 5)) more"
    fi
else
    echo "  ✗ No output files created"
fi

echo ""
echo "Spain load data preprocessing complete!"
echo "Output files saved to: $OUTPUT_DIR"
echo "You can now run Spain load experiments by updating your config to use:"
echo "  data.base_path: 'src/TS_MTL/data/spain_mf'"
echo "  data.sites: [city1, city2, city3, ...]"
