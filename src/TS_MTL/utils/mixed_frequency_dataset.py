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


class MixedFrequencyDataset(Dataset):
    """Dataset for mixed frequency data, optimized for LF prediction only."""
    def __init__(self, hf_data, lf_data, hf_lookback, lf_lookback, forecast_horizon, freq_ratio, client_id=None):
        """
        Initialize the dataset.
        Args:
            hf_data: High-frequency data array (time_steps, features)
            lf_data: Low-frequency data array (time_steps, features)
            hf_lookback: Number of high-frequency lags to use for prediction
            lf_lookback: Number of low-frequency lags to use for prediction
            forecast_horizon: Number of periods to forecast
            freq_ratio: Ratio between high and low frequency
            client_id: Optional client identifier for multi-task learning
        """
        self.hf_data = hf_data
        self.lf_data = lf_data
        self.hf_lookback = hf_lookback
        self.lf_lookback = lf_lookback
        self.forecast_horizon = forecast_horizon
        self.freq_ratio = freq_ratio
        self.client_id = client_id
        
        # Calculate the number of samples
        self.num_samples = min(
            (len(hf_data) - hf_lookback - forecast_horizon * freq_ratio) // freq_ratio,
            len(lf_data) - lf_lookback - forecast_horizon
        )
    
    def __len__(self):
        """Get the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Returns:
            Tuple containing:
            - hf_encoder_input: High-frequency input for encoder
            - lf_decoder_input: Low-frequency input for decoder
            - lf_target: Low-frequency target values
            - client_id: Client identifier (if provided)
        """
        # Calculate indices for high-frequency data
        hf_encoder_start = idx * self.freq_ratio
        hf_encoder_end = hf_encoder_start + self.hf_lookback
        
        # Calculate indices for low-frequency data
        lf_decoder_start = idx
        lf_decoder_end = lf_decoder_start + self.lf_lookback
        lf_target_idx = lf_decoder_end + (self.forecast_horizon - 1)
        
        # Extract data
        hf_encoder_input = torch.tensor(self.hf_data[hf_encoder_start:hf_encoder_end], dtype=torch.float32)
        lf_decoder_input = torch.tensor(self.lf_data[lf_decoder_start:lf_decoder_end], dtype=torch.float32)
        lf_target = torch.tensor(self.lf_data[lf_target_idx:lf_target_idx+1], dtype=torch.float32)
        
        # Include client_id in the output if provided
        if self.client_id is not None:
            client_id_tensor = torch.tensor(self.client_id, dtype=torch.long)
            return hf_encoder_input, lf_decoder_input, lf_target, client_id_tensor
        
        return hf_encoder_input, lf_decoder_input, lf_target
