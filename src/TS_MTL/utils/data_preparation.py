import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional
import time
import json
import dateutil
from datetime import datetime

from .custom_scaler import CustomScaler
from .mixed_frequency_dataset import MixedFrequencyDataset

#####################################################
# Data Preparation Functions
#####################################################

def prepare_client_data(
    hf_file, lf_file, features, target, min_date=None, max_date=None, 
    freq_ratio=90, debug=False
):
    """Prepare data for a single client from CSV files."""
    if debug:
        print(f"Loading data for client: {os.path.basename(hf_file)}")
    
    # Load data
    hf_df = pd.read_csv(hf_file, index_col=0, parse_dates=True)
    lf_df = pd.read_csv(lf_file, index_col=0, parse_dates=True)
    
    # Sort by date
    hf_df = hf_df.sort_index()
    lf_df = lf_df.sort_index()
    
    # Filter Features
    hf_df = hf_df[features]
    if isinstance(target, list):
        lf_df = lf_df[target]
    else:
        lf_df = lf_df[[target]]
    
    # Filter data between min_date and max_date
    if min_date:
        min_date = dateutil.parser.parse(min_date) if isinstance(min_date, str) else min_date
        hf_df = hf_df[hf_df.index >= min_date]
        lf_df = lf_df[lf_df.index >= min_date]
        
    if max_date:
        max_date = dateutil.parser.parse(max_date) if isinstance(max_date, str) else max_date
        hf_df = hf_df[hf_df.index <= max_date]
        lf_df = lf_df[lf_df.index <= max_date]
    
    # Handle missing values
    hf_df = hf_df.dropna()
    lf_df = lf_df.dropna()
    
    # Calculate statistics for scaling
    hf_mean, hf_std = hf_df.mean(), hf_df.std()
    lf_mean, lf_std = lf_df.mean(), lf_df.std()
    
    # Fix zero standard deviations
    hf_std = hf_std.replace(0, 1)
    lf_std = lf_std.replace(0, 1)
    
    # Apply scaling
    hf_scaled = (hf_df - hf_mean) / (hf_std + 1e-8)
    lf_scaled = (lf_df - lf_mean) / (lf_std + 1e-8)
    
    # Create custom scaler instances
    hf_scaler = CustomScaler(hf_mean, hf_std)
    lf_scaler = CustomScaler(lf_mean, lf_std)
    
    # Convert to NumPy arrays
    hf_data = hf_scaled.values
    lf_data = lf_scaled.values
    
    if debug:
        print(f"Final data shapes: HF: {hf_data.shape}, LF: {lf_data.shape}")
    
    return hf_data, lf_data, hf_scaler, lf_scaler, hf_df.columns, lf_df.columns


def create_client_datasets_with_id(
    client_data_pairs, features, target, client_ids=None, min_date=None, max_date=None,
    hf_lookback=192, lf_lookback=14, forecast_horizon=1, freq_ratio=90, train_ratio=0.8, debug=False
):
    """Create datasets for multiple clients with client IDs."""
    client_datasets = {}
    
    if client_ids is None:
        client_ids = [i+1 for i in range(len(client_data_pairs))]
    
    for i, ((hf_file, lf_file), client_id) in enumerate(zip(client_data_pairs, client_ids)):
        if debug:
            print(f"\nProcessing client {client_id}: {os.path.basename(hf_file)}")
        
        try:
            hf_data, lf_data, hf_scaler, lf_scaler, hf_cols, lf_cols = prepare_client_data(
                hf_file=hf_file,
                lf_file=lf_file,
                features=features,
                target=target,
                min_date=min_date,
                max_date=max_date,
                freq_ratio=freq_ratio,
                debug=debug
            )
            
            # Create dataset with client_id
            dataset = MixedFrequencyDataset(
                hf_data=hf_data,
                lf_data=lf_data,
                hf_lookback=hf_lookback,
                lf_lookback=lf_lookback,
                forecast_horizon=forecast_horizon,
                freq_ratio=freq_ratio,
                client_id=client_id
            )
            
            # Split into train and test
            train_size = int(train_ratio * len(dataset))
            test_size = len(dataset) - train_size
            
            train_indices = list(range(0, train_size))
            test_indices = list(range(train_size, len(dataset)))
            
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)
            
            client_datasets[client_id] = {
                'train_dataset': train_dataset,
                'test_dataset': test_dataset,
                'hf_scaler': hf_scaler,
                'lf_scaler': lf_scaler,
                'hf_cols': hf_cols,
                'lf_cols': lf_cols,
                'hf_data_shape': hf_data.shape,
                'lf_data_shape': lf_data.shape
            }
            
        except Exception as e:
            print(f"Error processing client data {client_id}: {e}")
            continue
    
    return client_datasets


def combine_client_datasets(client_datasets, mode='train'):
    """Combine all clients' train/test datasets."""
    datasets = [client_data[f'{mode}_dataset'] for client_id, client_data in client_datasets.items()]
    return ConcatDataset(datasets)