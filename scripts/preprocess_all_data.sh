#!/bin/bash

# Master Data Preprocessing Script
# This script runs all data preprocessors in sequence
#
# Usage:
#   ./scripts/preprocess_all_data.sh [dataset_number] [options]
#   Options:
#     --non-interactive    Run without user prompts (for SSH/batch execution)
#     --log-file <path>    Specify log file path (default: ./logs/preprocess_all.log)
#     --help              Show this help message
#   
#   Dataset numbers:
#     1 - Air Quality (Beijing Multi-Site)
#     2 - Load Forecasting (Electricity Load + Temperature)
#     3 - Wind Power Forecasting
#     4 - Spain Multi-Site Load
#     5 - All datasets

set -e  # Exit on any error

# Configuration
NON_INTERACTIVE=false
LOG_FILE=""
DATASET_CHOICE=""
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
            echo "TS-MTL Data Preprocessing Suite"
            echo "Usage: $0 [dataset_number] [options]"
            echo ""
            echo "Dataset numbers:"
            echo "  1 - Air Quality (Beijing Multi-Site)"
            echo "  2 - Load Forecasting (Electricity Load + Temperature)"
            echo "  3 - Wind Power Forecasting"
            echo "  4 - Spain Multi-Site Load"
            echo "  5 - All datasets"
            echo ""
            echo "Options:"
            echo "  --non-interactive    Run without user prompts (for SSH/batch execution)"
            echo "  --log-file <path>    Specify log file path"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 5 --non-interactive                    # Process all datasets in batch mode"
            echo "  $0 1 --log-file /tmp/air_quality.log     # Process air quality with custom log"
            echo "  ssh user@server 'cd project && $0 5 --non-interactive'  # Remote execution"
            exit 0
            ;;
        [1-5])
            if [[ -z "$DATASET_CHOICE" ]]; then
                DATASET_CHOICE="$1"
            else
                echo "Error: Multiple dataset choices specified"
                exit 1
            fi
            shift
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
    LOG_FILE="$PROJECT_ROOT/logs/preprocess_all.log"
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

log_message "=== TS-MTL Data Preprocessing Suite ==="
log_message "Project root: $PROJECT_ROOT"
log_message "Log file: $LOG_FILE"
log_message "Non-interactive mode: $NON_INTERACTIVE"

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if we're in the right directory
if [ ! -d "src/TS_MTL" ]; then
    log_error "TS-MTL project structure not found in $PROJECT_ROOT"
    log_error "Please ensure you're running this script from the TS-MTL project directory"
    exit 1
fi

# Make all preprocessing scripts executable
chmod +x scripts/preprocess_*.sh

# Prepare script arguments for non-interactive mode
SCRIPT_ARGS=""
if [ "$NON_INTERACTIVE" = true ]; then
    SCRIPT_ARGS="--non-interactive"
fi

log_message "Available datasets to process:"
log_message "1. Air Quality (Beijing Multi-Site)"
log_message "2. Load Forecasting (Electricity Load + Temperature)"
log_message "3. Wind Power Forecasting"
log_message "4. Spain Multi-Site Load"
log_message "5. All datasets"
log_message ""

# Determine choice - from command line or interactive
if [[ -n "$DATASET_CHOICE" ]]; then
    choice="$DATASET_CHOICE"
    log_message "Using dataset choice from command line: $choice"
elif [ "$NON_INTERACTIVE" = true ]; then
    log_error "Non-interactive mode requires a dataset choice (1-5)"
    log_message "Usage: $0 [1-5] --non-interactive"
    exit 1
else
    log_message "Running in interactive mode..."
    echo "Select dataset to process (1-5): "
    read -r choice
fi

# Function to run a preprocessing script with error handling
run_preprocessor() {
    local script_name="$1"
    local dataset_name="$2"
    local script_path="./scripts/$script_name"
    
    log_message "Processing $dataset_name dataset..."
    log_message "Running: $script_path $SCRIPT_ARGS"
    
    if [ ! -f "$script_path" ]; then
        log_error "Preprocessing script not found: $script_path"
        return 1
    fi
    
    if "$script_path" $SCRIPT_ARGS; then
        log_message "✓ $dataset_name preprocessing completed successfully"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 2 ]; then
            log_message "⚠ $dataset_name preprocessing completed with warnings"
            return 0
        else
            log_error "✗ $dataset_name preprocessing failed with exit code $exit_code"
            return 1
        fi
    fi
}

case $choice in
    1)
        run_preprocessor "preprocess_air_quality.sh" "Air Quality"
        ;;
    2)
        run_preprocessor "preprocess_load.sh" "Load Forecasting"
        ;;
    3)
        run_preprocessor "preprocess_wind.sh" "Wind Power"
        ;;
    4)
        run_preprocessor "preprocess_spain.sh" "Spain Multi-Site Load"
        ;;
    5)
        log_message "Processing all datasets in sequence..."
        failed_datasets=()
        successful_datasets=()
        
        for dataset_info in "preprocess_air_quality.sh:Air Quality" "preprocess_load.sh:Load Forecasting" "preprocess_wind.sh:Wind Power" "preprocess_spain.sh:Spain Multi-Site Load"; do
            IFS=':' read -r script_name dataset_name <<< "$dataset_info"
            
            if run_preprocessor "$script_name" "$dataset_name"; then
                successful_datasets+=("$dataset_name")
            else
                failed_datasets+=("$dataset_name")
                if [ "$NON_INTERACTIVE" = false ]; then
                    echo ""
                    read -p "Continue with remaining datasets? (y/N): " -n 1 -r
                    echo ""
                    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                        log_message "User chose to stop processing."
                        break
                    fi
                fi
            fi
        done
        
        # Summary for "all datasets" option
        log_message ""
        log_message "=== PROCESSING SUMMARY ==="
        log_message "Successful datasets (${#successful_datasets[@]}): ${successful_datasets[*]}"
        if [ ${#failed_datasets[@]} -gt 0 ]; then
            log_error "Failed datasets (${#failed_datasets[@]}): ${failed_datasets[*]}"
        fi
        
        if [ ${#failed_datasets[@]} -gt 0 ]; then
            exit 1
        fi
        ;;
    *)
        log_error "Invalid choice: $choice"
        log_message "Please select a number between 1-5"
        exit 1
        ;;
esac

log_message ""
log_message "Data preprocessing completed!"
log_message "Log file saved to: $LOG_FILE"
log_message ""
log_message "Next steps:"
log_message "  - Review the preprocessing logs above"
log_message "  - Check the output data files in src/TS_MTL/data/"
log_message "  - Run experiments using the run_*.sh scripts"
