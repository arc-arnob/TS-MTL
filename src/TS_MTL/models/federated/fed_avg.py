"""
Standard Federated Averaging (FedAvg) Implementation

This module contains the standard FedAvg algorithm implementation.
"""

from typing import Dict, List, Optional
from torch.utils.data import DataLoader
import logging

from .base_federated import BaseFederatedSystem

logger = logging.getLogger(__name__)


class FederatedAVGSystem(BaseFederatedSystem):
    """Standard Federated Averaging implementation."""
    
    def train_round(self, client_data_loaders: List[DataLoader], 
                   local_epochs: int = 1,
                   client_weights: Optional[List[float]] = None) -> Dict:
        """
        Perform one round of FedAvg training.
        Args:
            client_data_loaders: List of DataLoaders for each client
            local_epochs: Number of local epochs for each client
            client_weights: Optional weights for averaging (default: equal weights)
        Returns:
            Dictionary with training metrics
        """
        # Set models to training mode
        for model in self.client_models:
            model.train()
        
        # Track metrics
        metrics = {
            'losses': [0.0] * self.num_clients,
            'batch_count': [0] * self.num_clients
        }
        
        # Local training for each client
        for client_idx, data_loader in enumerate(client_data_loaders):
            client_loss = 0.0
            client_batches = 0
            
            for epoch in range(local_epochs):
                epoch_loss = 0.0
                epoch_batches = 0
                
                for batch in data_loader:
                    # Get batch data
                    hf_input, lf_input, lf_target = self.to_device(*batch)
                    
                    # Forward pass
                    self.client_optimizers[client_idx].zero_grad()
                    outputs = self.client_models[client_idx](hf_input, lf_input)
                    loss = self.criterion(outputs["lf_pred"], lf_target)
                    
                    # Backward pass
                    loss.backward()
                    self.client_optimizers[client_idx].step()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    epoch_batches += 1
                
                # Accumulate metrics across epochs
                if epoch_batches > 0:
                    client_loss += epoch_loss / epoch_batches
                client_batches += 1
            
            # Store average metrics
            if client_batches > 0:
                metrics['losses'][client_idx] = client_loss / client_batches
            metrics['batch_count'][client_idx] = client_batches
        
        # Perform federated averaging
        self._federated_averaging(client_weights)
        
        return metrics
