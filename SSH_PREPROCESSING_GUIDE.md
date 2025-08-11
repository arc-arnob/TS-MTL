# SSH Data Preprocessing Guide for TS-MTL

This guide explains how to run TS-MTL data preprocessing remotely via SSH, making it easy to prepare datasets on remote servers or clusters.

## Quick Start

### 1. System Requirements Check
Before processing data, verify your system is ready:

```bash
# Local check
./scripts/ssh_preprocess.sh check

# Remote check via SSH
ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh check'
```

### 2. Process All Datasets (Recommended)
Process all available datasets in non-interactive mode:

```bash
# Local processing
./scripts/ssh_preprocess.sh all --non-interactive

# Remote processing via SSH
ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh all --non-interactive'
```

### 3. Process Specific Dataset
Process individual datasets as needed:

```bash
# Air quality data
ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh air_quality --non-interactive'

# Load forecasting data
ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh load --non-interactive'

# Wind power data  
ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh wind --non-interactive'

# Spain multi-site load data
ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh spain --non-interactive'
```

## Detailed Usage

### SSH Preprocessing Wrapper

The `ssh_preprocess.sh` script provides a unified interface for all preprocessing operations:

```bash
Usage: ./scripts/ssh_preprocess.sh [dataset] [options]

Datasets:
  air_quality, air-quality, 1    Process Beijing air quality data
  load, load_forecasting, 2      Process electricity load forecasting data
  wind, wind_power, 3            Process wind power forecasting data
  spain, spain_load, 4           Process Spain multi-site load data
  all, 5                         Process all datasets
  check, verify                  Check system requirements

Options:
  --non-interactive, -n          Run without user prompts (recommended for SSH)
  --log-file <path>, -l <path>   Specify log file path
  --verbose, -v                  Enable verbose output
  --help, -h                     Show help message
```

### Individual Preprocessing Scripts

Each dataset has its own preprocessing script with enhanced SSH support:

- `preprocess_air_quality.sh` - Beijing air quality data
- `preprocess_load.sh` - Electricity load forecasting data
- `preprocess_wind.sh` - Wind power forecasting data
- `preprocess_spain.sh` - Spain multi-site load data
- `preprocess_all_data.sh` - Master script for all datasets

All scripts support:
```bash
./scripts/preprocess_*.sh [options]
  --non-interactive    Run without user prompts
  --log-file <path>    Specify log file path
  --help              Show help message
```

## Prerequisites

### Raw Data Files

Before running preprocessing, ensure you have the required raw data files:

1. **Air Quality Data** - Place in `./raw_data/air_quality/`:
   - `PRSA_Data_Aotizhongxin.csv`
   - `PRSA_Data_Dingling.csv`
   - `PRSA_Data_Gucheng.csv`
   - `PRSA_Data_Huairou.csv`
   - `PRSA_Data_Tiantan.csv`
   - `PRSA_Data_Wanshouxigong.csv`
   
   Source: [Beijing Multi-Site Air-Quality Data](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data)

2. **Load Forecasting Data** - Place in `./raw_data/energy/`:
   - `Load_history.csv`
   - `Temperature_history.csv`
   
   Source: [Global Energy Forecasting Competition 2012](https://www.kaggle.com/c/global-energy-forecasting-competition-2012-load-forecasting)

3. **Wind Power Data** - Place in `./raw_data/wind/`:
   - `train.csv`
   - `windforecasts_wf1.csv` through `windforecasts_wf7.csv`
   
   Source: Wind power forecasting competition data

4. **Spain Multi-Site Load Data** - Place in `./raw_data/spain/`:
   - `df_train.csv`
   
   Source: Spain electricity load forecasting data

### Python Environment

Required Python packages:
- pandas
- numpy

Install with:
```bash
pip install pandas numpy
```

## SSH Execution Examples

### Complete Remote Setup and Processing

```bash
# 1. Check system requirements
ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh check'

# 2. Process all datasets (recommended)
ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh all --non-interactive'

# 3. Verify output files
ssh user@server 'ls -la /path/to/TS-MTL/src/TS_MTL/data/'
```

### Processing with Custom Logging

```bash
# Process with custom log location
ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh all -n -l /tmp/preprocessing.log'

# Copy log file locally
scp user@server:/tmp/preprocessing.log ./
```

### Batch Processing Multiple Datasets

```bash
# Process datasets sequentially
for dataset in air_quality load wind spain; do
    echo "Processing $dataset..."
    ssh user@server "cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh $dataset --non-interactive"
done
```

### Background Processing

```bash
# Run preprocessing in background on remote server
ssh user@server 'cd /path/to/TS-MTL && nohup ./scripts/ssh_preprocess.sh all --non-interactive > /tmp/preprocess.out 2>&1 &'

# Check progress
ssh user@server 'tail -f /tmp/preprocess.out'
```

## Output Structure

After successful preprocessing, you'll find the mixed-frequency data files in:

```
src/TS_MTL/data/
├── air_quality/
│   ├── station-1-hf.csv    # High-frequency (15-min) features
│   ├── station-1-lf.csv    # Low-frequency (1-hour) target (CO)
│   ├── station-2-hf.csv
│   ├── station-2-lf.csv
│   └── ... (6 stations total)
├── load/
│   ├── zone-1-hf.csv       # High-frequency (15-min) temperature
│   ├── zone-1-lf.csv       # Low-frequency (1-hour) load
│   └── ... (20 zones total)
├── wind/
│   ├── wind-farm-1-hf.csv  # High-frequency (6-hour) weather
│   ├── wind-farm-1-lf.csv  # Low-frequency (daily) power
│   └── ... (7 farms total)
└── spain_mf/
    ├── city1.csv           # Multi-site load data
    └── ... (multiple cities)
```

## Logging and Monitoring

### Log Files

All scripts create detailed log files in the `logs/` directory:
- `logs/ssh_preprocess_YYYYMMDD_HHMMSS.log` - SSH wrapper logs
- `logs/preprocess_air_quality.log` - Air quality preprocessing logs
- `logs/preprocess_load.log` - Load forecasting preprocessing logs
- `logs/preprocess_wind.log` - Wind power preprocessing logs
- `logs/preprocess_spain.log` - Spain load preprocessing logs
- `logs/preprocess_all.log` - Master script logs

### Monitoring Progress

```bash
# Monitor real-time progress
ssh user@server 'tail -f /path/to/TS-MTL/logs/ssh_preprocess_*.log'

# Check for errors
ssh user@server 'grep ERROR /path/to/TS-MTL/logs/*.log'

# Check completion status
ssh user@server 'grep "SUCCESS\|COMPLETED" /path/to/TS-MTL/logs/*.log'
```

## Troubleshooting

### Common Issues

1. **Missing raw data files**
   - Error: "Missing X raw data files"
   - Solution: Download and place raw data files in the expected directories

2. **Python import errors**
   - Error: "IMPORT_ERROR: No module named 'pandas'"
   - Solution: Install required packages with `pip install pandas numpy`

3. **Permission errors**
   - Error: "Permission denied"
   - Solution: Ensure scripts are executable with `chmod +x scripts/*.sh`

4. **Path issues**
   - Error: "TS-MTL project structure not found"
   - Solution: Ensure you're running scripts from the TS-MTL root directory

### Debug Mode

Run with verbose output for debugging:

```bash
ssh user@server 'cd /path/to/TS-MTL && ./scripts/ssh_preprocess.sh all --non-interactive --verbose'
```

### Manual Verification

Verify preprocessing output manually:

```bash
# Check output file counts
ssh user@server 'find /path/to/TS-MTL/src/TS_MTL/data -name "*.csv" | wc -l'

# Check file sizes (should be non-zero)
ssh user@server 'find /path/to/TS-MTL/src/TS_MTL/data -name "*.csv" -size 0'

# Verify data content
ssh user@server 'head -5 /path/to/TS-MTL/src/TS_MTL/data/air_quality/station-1-hf.csv'
```

## Integration with TS-MTL Experiments

After preprocessing, the data is ready for use with TS-MTL experiments:

```bash
# Run experiments after preprocessing
ssh user@server 'cd /path/to/TS-MTL && ./scripts/run_all_experiments.sh'

# Or run specific experiments
ssh user@server 'cd /path/to/TS-MTL && ./scripts/run_ts_mtl_grad_bal.sh'
```

## Performance Notes

- **Air Quality**: ~6 station files, typically completes in 1-2 minutes
- **Load Forecasting**: ~20 zone files, typically completes in 3-5 minutes  
- **Wind Power**: ~7 farm files, typically completes in 1-2 minutes
- **Spain Load**: Multiple city files, typically completes in 1-2 minutes
- **All Datasets**: Total processing time typically 5-10 minutes

Processing time depends on data size and system performance. SSH execution may be slightly slower due to network latency in logging.
