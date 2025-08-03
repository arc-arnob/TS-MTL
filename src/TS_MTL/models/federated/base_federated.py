"""
Base Federated Learning System

This module contains the base classes and common functionality for federated learning systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-8
MAPE_EPSILON = 1e-10
DEFAULT_LEARNING_RATE = 0.001


class MetricsCalculator:
    """Utility class for calculating evaluation metrics."""
    
    @staticmethod
    def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate standard regression metrics."""
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean(np.square(predictions - targets))
        
        # Calculate MAPE with safeguard against division by zero
        mape = np.mean(np.abs((targets - predictions) / (np.abs(targets) + MAPE_EPSILON))) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'mape': mape
        }


class BaseFederatedSystem(ABC):
    """Base class for federated learning systems."""
    
    def __init__(self, client_models: List[nn.Module], 
                 client_optimizers: Optional[List[optim.Optimizer]] = None,
                 device: Optional[str] = None):
        
        # Device setup
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Store models and configuration
        self.client_models = client_models
        self.num_clients = len(client_models)
        
        # Move models to device
        for model in self.client_models:
            model.to(self.device)
        
        # Set up optimizers
        if client_optimizers is None:
            self.client_optimizers = [
                optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)
                for model in self.client_models
            ]
        else:
            self.client_optimizers = client_optimizers
        
        # Loss function and metrics
        self.criterion = nn.MSELoss()
        self.metrics_calc = MetricsCalculator()
    
    def to_device(self, *tensors):
        """Move tensors to the appropriate device."""
        return [tensor.to(self.device) for tensor in tensors]
    
    @abstractmethod
    def train_round(self, client_data_loaders: List[DataLoader], **kwargs) -> Dict:
        """Train one round of federated learning."""
        pass
    
    def evaluate(self, client_data_loaders: List[DataLoader]) -> List[Dict[str, float]]:
        """Evaluate all client models."""
        # Set models to evaluation mode
        for model in self.client_models:
            model.eval()
        
        client_metrics = []
        
        with torch.no_grad():
            for client_idx, data_loader in enumerate(client_data_loaders):
                # Collect predictions and targets
                all_preds = []
                all_targets = []
                total_loss = 0.0
                batch_count = 0
                
                for batch in data_loader:
                    # Unpack batch data
                    hf_input, lf_input, lf_target = self.to_device(*batch)
                    
                    # Forward pass
                    outputs = self.client_models[client_idx](hf_input, lf_input)
                    preds = outputs["lf_pred"]
                    
                    # Calculate loss
                    loss = self.criterion(preds, lf_target)
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # Store predictions and targets
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(lf_target.cpu().numpy())
                
                # Calculate metrics
                if batch_count > 0:
                    avg_loss = total_loss / batch_count
                    
                    # Concatenate all predictions and targets
                    all_preds = np.concatenate(all_preds, axis=0)
                    all_targets = np.concatenate(all_targets, axis=0)
                    
                    # Calculate additional metrics
                    metrics = self.metrics_calc.calculate_metrics(all_preds, all_targets)
                    metrics['loss'] = avg_loss
                    
                    client_metrics.append(metrics)
                else:
                    # Handle case with no data
                    client_metrics.append({
                        'loss': float('nan'),
                        'mae': float('nan'),
                        'mse': float('nan'),
                        'mape': float('nan')
                    })
        
        return client_metrics
    
    def predict(self, client_idx: int, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        """Generate predictions for a specific client."""
        model = self.client_models[client_idx]
        model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                hf_input, lf_input, lf_target = self.to_device(*batch)
                
                outputs = model(hf_input, lf_input)
                preds = outputs["lf_pred"]
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(lf_target.cpu().numpy())
        
        return {
            'predictions': np.concatenate(all_preds, axis=0) if all_preds else np.array([]),
            'targets': np.concatenate(all_targets, axis=0) if all_targets else np.array([])
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'client_models': [model.state_dict() for model in self.client_models],
            'client_optimizers': [opt.state_dict() for opt in self.client_optimizers],
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, model in enumerate(self.client_models):
            model.load_state_dict(checkpoint['client_models'][i])
        
        for i, opt in enumerate(self.client_optimizers):
            opt.load_state_dict(checkpoint['client_optimizers'][i])
        
        logger.info(f"Checkpoint loaded from {path}")
    
    def _federated_averaging(self, client_weights: Optional[List[float]] = None):
        """Perform federated averaging of model parameters."""
        if client_weights is None:
            client_weights = [1.0 / self.num_clients] * self.num_clients
        
        # Get state dictionaries from all clients
        client_states = [model.state_dict() for model in self.client_models]
        
        # Initialize global state
        global_state = copy.deepcopy(client_states[0])
        for key in global_state.keys():
            global_state[key] = global_state[key] * 0.0
        
        # Weighted average
        for client_idx, client_state in enumerate(client_states):
            weight = client_weights[client_idx]
            for key in global_state.keys():
                global_state[key] += client_state[key] * weight
        
        # Update all client models
        for model in self.client_models:
            model.load_state_dict(global_state)
        
        logger.debug("Federated averaging completed")
