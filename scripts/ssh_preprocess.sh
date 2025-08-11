#!/bin/bash

# SSH-Ready Data Preprocessing Wrapper
# This script provides a robust interface for running data preprocessing via SSH
#
# Usage:
#   ./scripts/ssh_preprocess.sh [dataset] [options]
#
# Examples:
#   # Process all datasets remotely in non-interactive mode
#   ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh all --non-interactive'
#
#   # Process specific dataset with logging
#   ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh air_quality --log-file /tmp/air.log'
#
#   # Check system requirements
#   ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh check'

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NON_INTERACTIVE=false
LOG_FILE=""
DATASET=""
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        air_quality|air-quality|1)
            DATASET="air_quality"
            shift
            ;;
        load|load_forecasting|2)
            DATASET="load"
            shift
            ;;
        wind|wind_power|3)
            DATASET="wind"
            shift
            ;;
        spain|spain_load|4)
            DATASET="spain"
            shift
            ;;
        all|5)
            DATASET="all"
            shift
            ;;
        check|verify)
            DATASET="check"
            shift
            ;;
        --non-interactive|-n)
            NON_INTERACTIVE=true
            shift
            ;;
        --log-file|-l)
            LOG_FILE="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "SSH-Ready Data Preprocessing Wrapper"
            echo ""
            echo "Usage: $0 [dataset] [options]"
            echo ""
            echo "Datasets:"
            echo "  air_quality, air-quality, 1    Process Beijing air quality data"
            echo "  load, load_forecasting, 2      Process electricity load forecasting data"
            echo "  wind, wind_power, 3            Process wind power forecasting data"
            echo "  spain, spain_load, 4           Process Spain multi-site load data"
            echo "  all, 5                         Process all datasets"
            echo "  check, verify                  Check system requirements"
            echo ""
            echo "Options:"
            echo "  --non-interactive, -n          Run without user prompts (recommended for SSH)"
            echo "  --log-file <path>, -l <path>   Specify log file path"
            echo "  --verbose, -v                  Enable verbose output"
            echo "  --help, -h                     Show this help message"
            echo ""
            echo "SSH Examples:"
            echo "  # Process all datasets remotely"
            echo "  ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh all -n'"
            echo ""
            echo "  # Check if system is ready for processing"
            echo "  ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh check'"
            echo ""
            echo "  # Process with custom logging"
            echo "  ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh air_quality -n -l /tmp/air.log'"
            exit 0
            ;;
        *)
            echo "Unknown option or dataset: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Default to checking if no dataset specified
if [[ -z "$DATASET" ]]; then
    DATASET="check"
fi

# Set up logging
if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="$PROJECT_ROOT/logs/ssh_preprocess_$(date +%Y%m%d_%H%M%S).log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log messages
log_message() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    if [[ "$VERBOSE" == true ]] || [[ "$DATASET" == "check" ]]; then
        echo "$message" | tee -a "$LOG_FILE"
    else
        echo "$message" >> "$LOG_FILE"
        echo "$1"  # Only show the message without timestamp for cleaner output
    fi
}

# Function to log errors
log_error() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo "$message" | tee -a "$LOG_FILE" >&2
}

# Function to check system requirements
check_system() {
    log_message "=== System Requirements Check ==="
    log_message "Project root: $PROJECT_ROOT"
    log_message "Log file: $LOG_FILE"
    
    local errors=0
    
    # Check if we're in the right directory
    if [ ! -d "$PROJECT_ROOT/src/TS_MTL" ]; then
        log_error "TS-MTL project structure not found"
        errors=$((errors + 1))
    else
        log_message "✓ TS-MTL project structure found"
    fi
    
    # Check Python availability
    if ! command -v python &> /dev/null; then
        log_error "Python is not available in PATH"
        errors=$((errors + 1))
    else
        python_version=$(python --version 2>&1)
        log_message "✓ Python available: $python_version"
    fi
    
    # Check required Python packages
    log_message "Checking required Python packages..."
    local required_packages=("pandas" "numpy")
    for package in "${required_packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            log_message "✓ $package available"
        else
            log_error "$package not available - install with: pip install $package"
            errors=$((errors + 1))
        fi
    done
    
    # Check directory structure
    log_message "Checking directory structure..."
    local required_dirs=("src/TS_MTL/utils/data_preprocessors" "scripts")
    for dir in "${required_dirs[@]}"; do
        if [ -d "$PROJECT_ROOT/$dir" ]; then
            log_message "✓ $dir exists"
        else
            log_error "$dir missing"
            errors=$((errors + 1))
        fi
    done
    
    # Check preprocessing scripts
    log_message "Checking preprocessing scripts..."
    local scripts=("preprocess_air_quality.sh" "preprocess_load.sh" "preprocess_wind.sh" "preprocess_spain.sh" "preprocess_all_data.sh")
    for script in "${scripts[@]}"; do
        if [ -f "$PROJECT_ROOT/scripts/$script" ]; then
            if [ -x "$PROJECT_ROOT/scripts/$script" ]; then
                log_message "✓ $script exists and is executable"
            else
                log_message "⚠ $script exists but is not executable (will be fixed automatically)"
                chmod +x "$PROJECT_ROOT/scripts/$script"
            fi
        else
            log_error "$script missing"
            errors=$((errors + 1))
        fi
    done
    
    # Summary
    log_message ""
    if [ $errors -eq 0 ]; then
        log_message "✓ All system requirements satisfied - ready for data preprocessing"
        log_message ""
        log_message "To process data:"
        log_message "  ./scripts/ssh_preprocess.sh all --non-interactive"
        log_message ""
        log_message "Or use the individual scripts:"
        log_message "  ./scripts/preprocess_all_data.sh 5 --non-interactive"
        return 0
    else
        log_error "System requirements check failed with $errors errors"
        log_message "Please fix the above issues before running data preprocessing"
        return 1
    fi
}

# Function to run preprocessing based on dataset choice
run_preprocessing() {
    log_message "=== Running Data Preprocessing ==="
    log_message "Dataset: $DATASET"
    log_message "Non-interactive: $NON_INTERACTIVE"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Prepare arguments for preprocessing scripts
    local script_args=""
    if [ "$NON_INTERACTIVE" = true ]; then
        script_args="--non-interactive"
    fi
    
    case $DATASET in
        air_quality)
            log_message "Processing Air Quality dataset..."
            if [ -f "scripts/preprocess_air_quality.sh" ]; then
                ./scripts/preprocess_air_quality.sh $script_args
            else
                log_error "preprocess_air_quality.sh not found"
                return 1
            fi
            ;;
        load)
            log_message "Processing Load Forecasting dataset..."
            if [ -f "scripts/preprocess_load.sh" ]; then
                ./scripts/preprocess_load.sh $script_args
            else
                log_error "preprocess_load.sh not found"
                return 1
            fi
            ;;
        wind)
            log_message "Processing Wind Power dataset..."
            if [ -f "scripts/preprocess_wind.sh" ]; then
                ./scripts/preprocess_wind.sh $script_args
            else
                log_error "preprocess_wind.sh not found"
                return 1
            fi
            ;;
        spain)
            log_message "Processing Spain Multi-Site Load dataset..."
            if [ -f "scripts/preprocess_spain.sh" ]; then
                ./scripts/preprocess_spain.sh $script_args
            else
                log_error "preprocess_spain.sh not found"
                return 1
            fi
            ;;
        all)
            log_message "Processing all datasets..."
            if [ -f "scripts/preprocess_all_data.sh" ]; then
                ./scripts/preprocess_all_data.sh 5 $script_args
            else
                log_error "preprocess_all_data.sh not found"
                return 1
            fi
            ;;
        *)
            log_error "Unknown dataset: $DATASET"
            return 1
            ;;
    esac
}

# Main execution
main() {
    log_message "SSH-Ready Data Preprocessing Wrapper"
    log_message "Starting at $(date)"
    
    if [ "$DATASET" = "check" ]; then
        check_system
    else
        # Run system check first
        if check_system; then
            log_message ""
            run_preprocessing
        else
            log_error "System requirements check failed - aborting preprocessing"
            exit 1
        fi
    fi
    
    log_message ""
    log_message "Completed at $(date)"
    log_message "Log saved to: $LOG_FILE"
}

# Run main function
main
