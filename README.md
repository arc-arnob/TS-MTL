# TS-MTL: Time Series Multi-Task Learning

This repository implements multi-task learning approaches for mixed-frequency time series forecasting, comparing traditional MTL methods with diffusion-based and Transformer based approaches across multiple gradient balancing techniques.

## Overview

The project compares **Multi-Task Learning (MTL)** with other techniques.

## Models & Methods

### Model Architectures
1. **Hard Parameter Sharing MTL**: Traditional multi-task learning with shared encoder-decoder and client-specific output layers
2. **TSDiff**: Diffusion-based time series generation model for multi-step forecasting
3. **ARIMAX Independent**: Separate ARIMAX models trained for each site/client
4. **ARIMAX Global**: Single ARIMAX model trained on pooled data from all sites

### Gradient Balancing Techniques
1. **Normal Training**: Standard gradient descent without conflict handling
2. **PCGrad**: Projects conflicting gradients onto normal planes to reduce interference
3. **CAGrad**: Conflict-Averse Gradient descent for balanced multi-task optimization
4. **Gradient Balancing**: Dynamic weighting based on cosine similarity and loss magnitudes

## Experimental Setup

### Dataset
- **Air Quality Data**: 6 monitoring stations (station-1 to station-6)
- **High-frequency features**: PM2.5, NO2, PM10 (15-minute intervals)
- **Low-frequency target**: CO (hourly intervals)
- **Time period**: September 1 - November 12, 2014

### Configuration
- **Lookback windows**: 64 (HF), 32 (LF)
- **Forecast horizon**: 1-16 steps
- **Frequency ratio**: 2:1 (HF:LF)
- **Train/test split**: 80/20
- **Batch size**: 16
- **Training epochs**: 30

## Installation

### Requirements
```bash
# Python 3.8+
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib
pip install hydra-core omegaconf
pip install scikit-learn
pip install statsmodels  # For ARIMAX baselines
pip install dateutil
```

### Setup
```bash
git clone <repository-url>
cd TS-MTL
pip install -e .
```

## Usage

### Quick Start

### Running Individual Experiments

**Multi-Task Learning experiments:**
```bash
# MTL with normal training
./scripts/run_ts_mtl_no_grad_bal.sh

# MTL with PCGrad  
./scripts/run_ts_mtl_pc_grad.sh

# MTL with CAGrad
./scripts/run_ts_mtl_ca_grad.sh

# MTL with Gradient Balancing
./scripts/run_ts_mtl_grad_bal.sh
```

**TSDiff experiments:**
```bash
# TSDiff baseline
./scripts/run_ts_diff.sh

# TSDiff with PCGrad
./scripts/run_ts_diff_pc_grad.sh

# TSDiff with CAGrad  
./scripts/run_ts_diff_ca_grad.sh

# TSDiff with Gradient Balancing
./scripts/run_ts_diff_grad_bal.sh
```

**ARIMAX baselines:**
```bash
# Independent ARIMAX models (one per site)
./scripts/run_arimax_independent.sh

# Global ARIMAX model (pooled data)
./scripts/run_arimax_global.sh
```

**Run all experiments:**
```bash
./scripts/run_all_experiments.sh
```
Run individual experiments:
```bash
# MTL Experiments
./scripts/run_ts_mtl_no_grad_bal.sh    # Baseline MTL
./scripts/run_ts_mtl_pc_grad.sh        # MTL + PCGrad
./scripts/run_ts_mtl_ca_grad.sh        # MTL + CAGrad

# TSDiff Experiments  
./scripts/run_ts_diff.sh               # Baseline TSDiff
./scripts/run_ts_diff_pc_grad.sh       # TSDiff + PCGrad
./scripts/run_ts_diff_ca_grad.sh       # TSDiff + CAGrad
./scripts/run_ts_diff_grad_bal.sh      # TSDiff + Gradient Balancing
```

### Run All Experiments
```bash
./scripts/run_all_experiments.sh
```

### Custom Configuration
```bash
# Example: Custom MTL experiment
python -m TS_MTL.cli \
  model.name=hard_sharing \
  trainer.name=pc_grad_trainer \
  train.epochs=50 \
  data.batch_size=32 \
  trainer.params.learning_rate=0.0001
```

### Direct Python Usage
```python
from TS_MTL.cli import main
from omegaconf import DictConfig

# Configure experiment
cfg = DictConfig({
    "model": {"name": "ts_diff"},
    "trainer": {"name": "ts_diff_trainer"},
    "train": {"epochs": 30}
    # ... other parameters
})

# Run experiment
results = main(cfg)
```

## Experiment Overview

### Complete Experimental Matrix (9 experiments):

| Model | Trainer | Description |
|-------|---------|-------------|
| MTL | Normal | Baseline multi-task learning |
| MTL | PCGrad | MTL with projection-based conflict resolution |
| MTL | CAGrad | MTL with conflict-averse optimization |
| TSDiff | Normal | Baseline diffusion model |
| TSDiff | PCGrad | Diffusion with PCGrad |
| TSDiff | CAGrad | Diffusion with CAGrad |
| TSDiff | GradBal | Diffusion with dynamic gradient balancing |
| ARIMAX | Independent | Statistical baseline with separate models per site |
| ARIMAX | Global | Statistical baseline with pooled data across sites |

### Evaluation Metrics
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **MAPE**: Mean Absolute Percentage Error
- **Multi-horizon**: Evaluated at horizons [1, 3, 6, 12]

## Project Structure

```
TS-MTL/
├── src/TS_MTL/
│   ├── models/
│   │   ├── non_fed_baselines/
│   │   │   ├── ts_mtl.py          # MTL model implementations
│   │   │   ├── ts_diff.py         # Diffusion model implementations
│   │   │   ├── arimax_independent.py # Independent ARIMAX models
│   │   │   ├── arimax_global.py   # Global ARIMAX model
│   │   │   └── arimax_models.py   # ARIMAX model wrappers
│   │   └── federated/             # Federated learning models
│   ├── trainers/
│   │   ├── ts_mtl/            # MTL trainers
│   │   │   ├── normal_trainer.py
│   │   │   ├── pc_grad_trainer.py
│   │   │   └── ca_grad_trainer.py
│   │   └── ts_diff/           # TSDiff trainers
│   │       ├── ts_diff_trainer.py
│   │       ├── pc_grad_trainer.py
│   │       ├── ca_grad_trainer.py
│   │       └── grad_bal_trainer.py
│   ├── utils/
│   │   ├── data_preparation.py
│   │   ├── mixed_frequency_dataset.py
│   │   └── custom_scaler.py
│   └── cli.py                 # Main CLI interface
├── scripts/
│   ├── run_ts_mtl_*.sh        # MTL experiment scripts
│   ├── run_ts_diff_*.sh       # TSDiff experiment scripts
│   ├── run_arimax_*.sh        # ARIMAX baseline scripts
│   └── run_all_experiments.sh # Batch execution
│   ├── run_ts_diff_*.sh       # TSDiff experiment scripts
│   └── run_all_experiments.sh # Batch execution
├── configs/
│   └── default.yaml           # Default configuration
└── data/
    └── air_quality/           # Dataset files
```

## Configuration

The system uses Hydra for configuration management. Key parameters:

```yaml
# Model Configuration
model:
  name: "hard_sharing"  # or "ts_diff", "arimax_independent", "arimax_global"
  params:
    hidden_dim: 64
    num_layers: 2
    dropout: 0.2
    # ARIMAX-specific parameters
    lookback_days: 32
    save_plots: false

# Trainer Configuration  
trainer:
  name: "normal_trainer"  # or pc_grad_trainer, etc.
  params:
    learning_rate: 0.001
    device: "cpu"

# Data Configuration
data:
  base_path: "src/TS_MTL/data/air_quality"
  features: ["PM2.5", "NO2", "PM10"]
  target: ["CO"]
  batch_size: 16
  train_ratio: 0.8

# Training Configuration
train:
  epochs: 30
  eval_horizons: [1, 3, 6, 12]
```

## Output

Results are logged to:
- **Console**: Training progress and evaluation metrics
- **Files**: Predictions and targets logged to `preds_targets_log.txt`
- **Hydra outputs**: Experiment configurations and logs in `outputs/`


## Research Applications

This codebase supports research in:
- **Multi-task learning** for time series
- **Confidentiality** for distributed data sources
- **Diffusion models** for forecasting
- **Gradient conflict resolution** in multi-client scenarios  
- **Mixed-frequency** time series modeling
- **Multi-horizon** forecasting evaluation

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{ts-mtl-2024,
  title={Time Series Multi-Task Learning with Diffusion Models},
  author={Arnob Chowdhury},
  year={2024},
  url={}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
