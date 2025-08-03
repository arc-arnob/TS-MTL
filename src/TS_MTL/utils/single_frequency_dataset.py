import torch
import numpy as np
from torch.utils.data import Dataset

class SingleFrequencyDataset(Dataset):
    """Dataset for mixed frequency data - interpolates HF to LF rate for multivariate forecasting."""
    def __init__(self, hf_data, lf_data, hf_lookback, lf_lookback, forecast_horizon, freq_ratio, client_id=None, mode='train'):
        """
        Initialize the dataset.
        Args:
            hf_data: High-frequency data array (time_steps, features)
            lf_data: Low-frequency data array (time_steps, features)
            hf_lookback: Number of high-frequency lags to use for prediction
            lf_lookback: Number of low-frequency lags to use for prediction
            forecast_horizon: Number of periods to forecast
            freq_ratio: Ratio between high and low frequency
            client_id: Optional client identifier
            mode: 'train' or 'test' mode
        """
        self.freq_ratio = freq_ratio
        self.lf_lookback = lf_lookback
        self.forecast_horizon = forecast_horizon
        self.client_id = client_id
        self.mode = mode
        
        # Interpolate HF data to match LF frequency
        self.hf_downsampled = self._downsample_hf_to_lf(hf_data, lf_data.shape[0])
        
        # Combine HF and LF data into multivariate time series
        self.combined_data = np.concatenate([self.hf_downsampled, lf_data], axis=1)
        self.lf_feature_indices = list(range(self.hf_downsampled.shape[1], self.combined_data.shape[1]))
        
        # Calculate the number of samples
        self.num_samples = len(self.combined_data) - lf_lookback - forecast_horizon
    
    def _downsample_hf_to_lf(self, hf_data, target_length):
        """Downsample high-frequency data to match low-frequency length."""
        if len(hf_data) <= target_length:
            # If HF data is not actually higher frequency, just pad or truncate
            if len(hf_data) < target_length:
                # Pad with last values
                padding = np.tile(hf_data[-1:], (target_length - len(hf_data), 1))
                return np.concatenate([hf_data, padding], axis=0)
            else:
                return hf_data[:target_length]
        
        # Simple linear interpolation/downsampling
        from scipy import interpolate
        
        # Create interpolation functions for each feature
        hf_indices = np.linspace(0, len(hf_data) - 1, len(hf_data))
        lf_indices = np.linspace(0, len(hf_data) - 1, target_length)
        
        downsampled = np.zeros((target_length, hf_data.shape[1]))
        
        for i in range(hf_data.shape[1]):
            f = interpolate.interp1d(hf_indices, hf_data[:, i], kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
            downsampled[:, i] = f(lf_indices)
        
        return downsampled
    
    def __len__(self):
        """Get the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Returns multivariate sequences with both HF and LF features.
        """
        # Calculate indices for the combined data
        start_idx = idx
        context_end = start_idx + self.lf_lookback
        target_end = context_end + self.forecast_horizon
        
        if self.mode == 'train':
            # For training, return the complete sequence (context + target)
            # Note: TSDiff will use the entire sequence for denoising training
            full_sequence = torch.tensor(self.combined_data[start_idx:target_end], dtype=torch.float32)
            return full_sequence, self.client_id if self.client_id is not None else 0
        else:
            # For testing, separate context and target
            # Context: multivariate (HF + LF features)
            context = torch.tensor(self.combined_data[start_idx:context_end], dtype=torch.float32)
            
            # Target: only LF features (what we want to forecast)
            target_multivariate = self.combined_data[context_end:target_end]
            target_lf_only = target_multivariate[:, self.lf_feature_indices]
            target = torch.tensor(target_lf_only, dtype=torch.float32)
            
            return context, target, self.client_id if self.client_id is not None else 0