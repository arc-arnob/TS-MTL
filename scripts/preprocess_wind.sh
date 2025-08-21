#!/bin/bash

# Wind Power Data Preprocessor Runner
# This script processes raw wind power and forecast data
#
# Usage:
#   ./scripts/preprocess_wind.sh [options]
#   Options:
#     --non-interactive    Run without user prompts (for SSH/batch execution)
#     --log-file <path>    Specify log file path (default: ./logs/preprocess_wind.log)
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
            echo "Wind Power Data Preprocessor"
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
    LOG_FILE="$PROJECT_ROOT/logs/preprocess_wind.log"
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

log_message "=== Wind Power Data Preprocessor ==="
log_message "Converting raw wind power and forecast data to mixed-frequency format..."
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
log_message "Creating output directories..."
mkdir -p ./src/TS_MTL/data/wind
mkdir -p ./raw_data/wind
mkdir -p ./logs

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

log_message "Checking for raw data files in $RAW_DIR..."
missing_files=0
for file in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "$RAW_DIR/$file" ]; then
        log_error "Missing: $RAW_DIR/$file"
        missing_files=$((missing_files + 1))
    else
        log_message "Found: $RAW_DIR/$file"
    fi
done

if [ $missing_files -gt 0 ]; then
    log_error "Missing $missing_files raw data files."
    log_message "Please download the wind power forecasting dataset and place the CSV files in:"
    log_message "  $RAW_DIR/"
    log_message ""
    log_message "Expected files:"
    for file in "${EXPECTED_FILES[@]}"; do
        log_message "  - $file"
    done
    log_message ""
    log_message "Dataset source: Wind power forecasting competition data (e.g., GEFCom2012 Wind Track)"
    
    if [ "$NON_INTERACTIVE" = false ]; then
        echo ""
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_message "User chose to abort."
            exit 1
        fi
        log_message "User chose to continue despite missing files."
    else
        log_error "Cannot continue in non-interactive mode with missing files."
        exit 1
    fi
fi

# Set PYTHONPATH for package importability
log_message "Setting up Python environment..."
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/src"

# Check if Python and required packages are available
if ! command -v python &> /dev/null; then
    log_error "Python is not available in PATH"
    exit 1
fi

# Run the preprocessor
log_message "Running wind power data preprocessor..."
log_message "PYTHONPATH: $PYTHONPATH"

# Use a more robust Python execution method
if python -c "
import sys
import os
sys.path.insert(0, '${PROJECT_ROOT}/src')

try:
    from TS_MTL.utils.data_preprocessors.multi_freq_data_processors.wind_preprocessor import main
    print('Starting wind power data preprocessing...')
    main()
    print('Wind power data preprocessing completed successfully.')
except Exception as e:
    print(f'PREPROCESSING_ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"; then
    log_message "Preprocessing completed successfully"
else
    log_error "Preprocessing failed"
    exit 1
fi

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
