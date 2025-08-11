#!/bin/bash

# Air Quality Data Preprocessor Runner
# This script processes raw air quality data and converts it to mixed-frequency format
# 
# Usage:
#   ./scripts/preprocess_air_quality.sh [options]
#   Options:
#     --non-interactive    Run without user prompts (for SSH/batch execution)
#     --log-file <path>    Specify log file path (default: ./logs/preprocess_air_quality.log)
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
            echo "Air Quality Data Preprocessor"
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
    LOG_FILE="$PROJECT_ROOT/logs/preprocess_air_quality.log"
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

log_message "=== Air Quality Data Preprocessor ==="
log_message "Converting raw air quality data to mixed-frequency format..."
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
mkdir -p ./src/TS_MTL/data/air_quality
mkdir -p ./raw_data/air_quality
mkdir -p ./logs

# Check if raw data exists
RAW_DIR="./raw_data/air_quality"
EXPECTED_FILES=(
    "PRSA_Data_Aotizhongxin.csv"
    "PRSA_Data_Dingling.csv"
    "PRSA_Data_Gucheng.csv"
    "PRSA_Data_Huairou.csv"
    "PRSA_Data_Tiantan.csv"
    "PRSA_Data_Wanshouxigong.csv"
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
    log_message "Please download the Beijing air quality dataset and place the CSV files in:"
    log_message "  $RAW_DIR/"
    log_message ""
    log_message "Expected files:"
    for file in "${EXPECTED_FILES[@]}"; do
        log_message "  - $file"
    done
    log_message ""
    log_message "Dataset source: https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data"
    
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

# Check if required packages are importable
python_check_result=$(python -c "
try:
    import pandas
    import numpy
    print('SUCCESS')
except ImportError as e:
    print(f'IMPORT_ERROR: {e}')
" 2>&1)

if [[ $python_check_result == IMPORT_ERROR* ]]; then
    log_error "Required Python packages not available: ${python_check_result#IMPORT_ERROR: }"
    log_message "Please install required packages: pip install pandas numpy"
    exit 1
fi

# Run the preprocessor
log_message "Running air quality data preprocessor..."
log_message "PYTHONPATH: $PYTHONPATH"

python_script_result=$(python -c "
import sys
import os
sys.path.insert(0, '${PROJECT_ROOT}/src')

try:
    from TS_MTL.utils.data_preprocessors.multi_freq_data_processors.air_quality_data_preprocessor import main
    print('Starting air quality data preprocessing...')
    main()
    print('Air quality data preprocessing completed successfully.')
except Exception as e:
    print(f'PREPROCESSING_ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1)

if [[ $python_script_result == *PREPROCESSING_ERROR* ]]; then
    log_error "Preprocessing failed: $python_script_result"
    exit 1
else
    log_message "Preprocessing output: $python_script_result"
fi

# Check if output files were created
OUTPUT_DIR="./src/TS_MTL/data/air_quality"
log_message "Verifying output files in $OUTPUT_DIR..."
output_success=true
created_files=()
missing_files=()

for i in {1..6}; do
    hf_file="$OUTPUT_DIR/station-$i-hf.csv"
    lf_file="$OUTPUT_DIR/station-$i-lf.csv"
    
    if [ -f "$hf_file" ] && [ -f "$lf_file" ]; then
        log_message "✓ Station $i files created successfully"
        created_files+=("station-$i-hf.csv" "station-$i-lf.csv")
    else
        log_error "✗ Station $i files missing"
        missing_files+=("station-$i-hf.csv" "station-$i-lf.csv")
        output_success=false
    fi
done

# Summary
log_message ""
if [ "$output_success" = true ]; then
    log_message "SUCCESS: Air quality data preprocessing completed successfully!"
    log_message "Created ${#created_files[@]} output files in: $OUTPUT_DIR"
else
    log_error "PARTIAL SUCCESS: Some output files are missing (${#missing_files[@]} files)"
    log_message "Created files: ${#created_files[@]}"
    log_message "Missing files: ${#missing_files[@]}"
fi

log_message ""
log_message "Next steps - You can now run air quality experiments with:"
log_message "  ./scripts/run_arimax_independent.sh"
log_message "  ./scripts/run_arimax_global.sh"
log_message "  ./scripts/run_ts_mtl_grad_bal.sh"
log_message ""
log_message "Log file saved to: $LOG_FILE"

# Exit with appropriate code
if [ "$output_success" = true ]; then
    exit 0
else
    exit 2  # Partial success
fi
