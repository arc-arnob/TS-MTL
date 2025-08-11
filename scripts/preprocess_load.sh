#!/bin/bash

# Load Forecasting Data Preprocessor Runner
# This script processes raw electricity load and temperature data
#
# Usage:
#   ./scripts/preprocess_load.sh [options]
#   Options:
#     --non-interactive    Run without user prompts (for SSH/batch execution)
#     --log-file <path>    Specify log file path (default: ./logs/preprocess_load.log)
#     --help              Show this help message

set -e  # Exit on any error

# Configuration
NON_INTERACTIVE=false
LOG_FILE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --non-interactive)
            NON_INTERACTIVE=true
            shift
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Load Forecasting Data Preprocessor"
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --non-interactive    Run without user prompts (for SSH/batch execution)"
            echo "  --log-file <path>    Specify log file path"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set up logging
if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="$PROJECT_ROOT/logs/preprocess_load.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log messages
log_message() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$LOG_FILE"
}

# Function to log errors
log_error() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo "$message" | tee -a "$LOG_FILE" >&2
}

log_message "=== Load Forecasting Data Preprocessor ==="
log_message "Converting raw load and temperature data to mixed-frequency format..."
log_message "Project root: $PROJECT_ROOT"
log_message "Log file: $LOG_FILE"

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if we're in the right directory
if [ ! -d "src/TS_MTL" ]; then
    log_error "TS-MTL project structure not found in $PROJECT_ROOT"
    log_error "Please ensure you're running this script from the TS-MTL project directory"
    exit 1
fi

# Create necessary directories
echo "Creating output directories..."
mkdir -p ./src/TS_MTL/data/load
mkdir -p ./raw_data/energy

# Check if raw data exists
RAW_DIR="./raw_data/energy"
EXPECTED_FILES=(
    "Load_history.csv"
    "Temperature_history.csv"
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
    echo "Please download the electricity load forecasting dataset and place the CSV files in:"
    echo "  $RAW_DIR/"
    echo ""
    echo "Expected files:"
    for file in "${EXPECTED_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Dataset source: GEFCom2012 or similar load forecasting competition data"
    exit 1
fi

# Set PYTHONPATH if TS_MTL package is not installed
if ! python -c "import TS_MTL" 2>/dev/null; then
    echo "Setting PYTHONPATH..."
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
fi

# Run the preprocessor
echo "Running load forecasting data preprocessor..."
python -c "
import sys
sys.path.append('src')
from TS_MTL.utils.data_preprocessors.load_preprocessor import main
main()
"

# Check if output files were created
OUTPUT_DIR="./src/TS_MTL/data/load"
echo "Checking output files..."
for i in {1..20}; do
    if [ -f "$OUTPUT_DIR/zone-$i-hf.csv" ] && [ -f "$OUTPUT_DIR/zone-$i-lf.csv" ]; then
        echo "  ✓ Zone $i files created"
    else
        echo "  ✗ Zone $i files missing"
    fi
done

echo ""
echo "Load forecasting data preprocessing complete!"
echo "Output files saved to: $OUTPUT_DIR"
echo "You can now run load forecasting experiments by updating your config to use:"
echo "  data.base_path: 'src/TS_MTL/data/load'"
echo "  data.sites: [zone-1, zone-2, ..., zone-20]"
