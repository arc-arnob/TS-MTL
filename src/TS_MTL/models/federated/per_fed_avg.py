"""
Personalized Federated Averaging Implementations

This module contains implementations of personalized federated learning algorithms
including PersonalizedFedAVGSystem and SimplePFedAvgSystem.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import time
import copy
import logging

from .fed_avg import FederatedAVGSystem
from .model_components import FedAVGModel, ProximalTerm, DPOptimizer, DPOptimizerWithProximal, SecureAggregation, PrivacyPreservingFedAVGModel

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
#####################################################
# Personalized Federated Learning Systems
#####################################################

class PersonalizedFedAVGSystem(FederatedAVGSystem):
    """System to manage federated learning with Personalized Federated Averaging (Per-FedAvg)."""
    def __init__(self, client_models, client_optimizers=None, meta_learning_rate=0.01, device=None):
        super().__init__(client_models, client_optimizers, device)
        self.meta_learning_rate = meta_learning_rate
        
        # Create personalized models for each client (these will be updated separately)
        self.personalized_models = [copy.deepcopy(model) for model in self.client_models]
        
        # Create optimizers for personalized models
        self.personalized_optimizers = [
            optim.Adam(model.parameters(), lr=0.001)
            for model in self.personalized_models
        ]
    
    def train_round(self, client_data_loaders, local_epochs=1, client_weights=None, personalization_steps=5):
        """
        Perform one round of personalized federated training.
        Args:
            client_data_loaders: List of DataLoaders for each client
            local_epochs: Number of local epochs for each client
            client_weights: Optional weights for averaging (default: equal weights)
            personalization_steps: Number of personalization steps to perform after global update
        Returns:
            Dictionary with training metrics
        """
        # First, perform standard FedAVG training
        metrics = super().train_round(client_data_loaders, local_epochs, client_weights)
        
        # Add personalization metrics
        metrics['personalized_losses'] = [0.0] * self.num_clients
        metrics['personalized_batch_count'] = [0] * self.num_clients
        
        # Step 3: Personalization for each client
        # We'll fine-tune the personalized models based on the global model
        for client_idx, data_loader in enumerate(client_data_loaders):
            # Update personalized model with the global model first
            self.personalized_models[client_idx].load_state_dict(
                self.client_models[client_idx].state_dict()
            )
            
            # Move the personalized model to the appropriate device
            self.personalized_models[client_idx].to(self.device)
            self.personalized_models[client_idx].train()
            
            # Personalization training
            personalized_loss = 0.0
            batch_count = 0
            
            # Create a subset of the data for personalization
            personalization_data = []
            for batch in data_loader:
                personalization_data.append(batch)
                if len(personalization_data) >= personalization_steps:
                    break
            
            # If we have data for personalization
            for batch in personalization_data:
                # Get batch data
                hf_input, lf_input, lf_target = [b.to(self.device) for b in batch]
                
                # Forward pass on personalized model
                self.personalized_optimizers[client_idx].zero_grad()
                outputs = self.personalized_models[client_idx](hf_input, lf_input)
                
                # Prediction loss
                loss = self.criterion(outputs["lf_pred"], lf_target)
                
                # Backward and optimize
                loss.backward()
                self.personalized_optimizers[client_idx].step()
                
                # Update metrics
                personalized_loss += loss.item()
                batch_count += 1
            
            # Store personalization metrics
            if batch_count > 0:
                metrics['personalized_losses'][client_idx] = personalized_loss / batch_count
            metrics['personalized_batch_count'][client_idx] = batch_count
        
        return metrics
    
    def meta_update(self, client_data_loaders, meta_batch_size=1):
        """
        Perform meta-learning update for personalization.
        Args:
            client_data_loaders: List of DataLoaders for each client
            meta_batch_size: Number of batches to use for meta-update
        Returns:
            Dictionary with meta-update metrics
        """
        # Ensure all client models are in training mode
        for model in self.client_models:
            model.train()
            
        metrics = {
            'meta_losses': [0.0] * self.num_clients,
            'meta_batch_count': [0] * self.num_clients
        }
        
        # For each client
        for client_idx, data_loader in enumerate(client_data_loaders):
            # Get meta-batches (support and query sets)
            support_batches = []
            query_batches = []
            
            batch_count = 0
            for batch in data_loader:
                if batch_count % 2 == 0:
                    support_batches.append(batch)
                else:
                    query_batches.append(batch)
                
                batch_count += 1
                if batch_count >= meta_batch_size * 2:  # We need pairs of support/query
                    break
            
            # Skip if we don't have enough data
            if len(support_batches) == 0 or len(query_batches) == 0:
                continue
            
            # Clone the current model for temporary updates
            temp_model = copy.deepcopy(self.client_models[client_idx])
            temp_model.to(self.device)
            # Set model to training mode explicitly - this is crucial for RNN/LSTM backward pass
            temp_model.train()
            temp_optimizer = optim.SGD(temp_model.parameters(), lr=self.meta_learning_rate)
            
            meta_loss = 0.0
            meta_batch_count = 0
            
            # For each pair of support/query batches
            for support_batch, query_batch in zip(support_batches, query_batches):
                # Unpack support batch
                hf_input, lf_input, lf_target = [b.to(self.device) for b in support_batch]
                
                # Forward pass on support set
                temp_optimizer.zero_grad()
                outputs = temp_model(hf_input, lf_input)
                
                # Support loss
                support_loss = self.criterion(outputs["lf_pred"], lf_target)
                
                # Backward and optimize (simulate personalization)
                support_loss.backward()
                temp_optimizer.step()
                
                # Now evaluate on query set
                hf_input, lf_input, lf_target = [b.to(self.device) for b in query_batch]
                
                # Switch to evaluation mode for the query forward pass
                temp_model.eval()
                # Forward pass on query set with updated model
                with torch.enable_grad():  # Ensure we can still compute gradients
                    outputs = temp_model(hf_input, lf_input)
                # Switch back to training for backprop
                temp_model.train()
                
                # Query loss
                query_loss = self.criterion(outputs["lf_pred"], lf_target)
                
                # Update the original model based on query loss
                # This is the key step for meta-learning
                self.client_optimizers[client_idx].zero_grad()
                query_loss.backward()
                self.client_optimizers[client_idx].step()
                
                # Update metrics
                meta_loss += query_loss.item()
                meta_batch_count += 1
            
            # Store meta-update metrics
            if meta_batch_count > 0:
                metrics['meta_losses'][client_idx] = meta_loss / meta_batch_count
            metrics['meta_batch_count'][client_idx] = meta_batch_count
        
        # After meta-updates, perform federated averaging
        client_weights = [1.0 / self.num_clients] * self.num_clients
        self._federated_averaging(client_weights)
        
        return metrics
    
    def evaluate(self, client_data_loaders, use_personalized=True):
        """
        Evaluate models for all clients.
        Args:
            client_data_loaders: List of DataLoaders for each client
            use_personalized: Whether to use personalized models for evaluation
        Returns:
            Dictionary containing evaluation metrics for each client
        """
        # Set models to evaluation mode
        for model in self.client_models:
            model.eval()
        
        for model in self.personalized_models:
            model.eval()
        
        metrics = {
            'losses': [0.0] * self.num_clients,
            'personalized_losses': [0.0] * self.num_clients,
            'batch_count': [0] * self.num_clients,
            'all_preds': [[] for _ in range(self.num_clients)],
            'all_targets': [[] for _ in range(self.num_clients)],
            'all_personalized_preds': [[] for _ in range(self.num_clients)]
        }
        
        with torch.no_grad():
            for client_idx, data_loader in enumerate(client_data_loaders):
                for batch in data_loader:
                    # Unpack batch data
                    hf_input, lf_input, lf_target = [b.to(self.device) for b in batch]
                    
                    # Forward pass on global model
                    global_outputs = self.client_models[client_idx](hf_input, lf_input)
                    global_preds = global_outputs["lf_pred"]
                    
                    # Forward pass on personalized model
                    personalized_outputs = self.personalized_models[client_idx](hf_input, lf_input)
                    personalized_preds = personalized_outputs["lf_pred"]
                    
                    # Calculate losses
                    global_loss = self.criterion(global_preds, lf_target)
                    personalized_loss = self.criterion(personalized_preds, lf_target)
                    
                    # Update metrics
                    metrics['losses'][client_idx] += global_loss.item()
                    metrics['personalized_losses'][client_idx] += personalized_loss.item()
                    metrics['batch_count'][client_idx] += 1
                    
                    # Store predictions and targets
                    metrics['all_preds'][client_idx].append(global_preds.cpu().numpy())
                    metrics['all_personalized_preds'][client_idx].append(personalized_preds.cpu().numpy())
                    metrics['all_targets'][client_idx].append(lf_target.cpu().numpy())
        
        # Calculate final metrics
        client_metrics = []
        
        for client_idx in range(self.num_clients):
            if metrics['batch_count'][client_idx] > 0:
                # Average losses
                avg_global_loss = metrics['losses'][client_idx] / metrics['batch_count'][client_idx]
                avg_personalized_loss = metrics['personalized_losses'][client_idx] / metrics['batch_count'][client_idx]
                
                # Concatenate predictions and targets
                global_preds = np.concatenate(metrics['all_preds'][client_idx], axis=0)
                personalized_preds = np.concatenate(metrics['all_personalized_preds'][client_idx], axis=0)
                all_targets = np.concatenate(metrics['all_targets'][client_idx], axis=0)
                
                # Global model metrics
                global_mae = np.mean(np.abs(global_preds - all_targets))
                global_mse = np.mean(np.square(global_preds - all_targets))
                
                # Personalized model metrics
                personalized_mae = np.mean(np.abs(personalized_preds - all_targets))
                personalized_mse = np.mean(np.square(personalized_preds - all_targets))
                
                # Calculate MAPE with safeguard against division by zero
                epsilon = 1e-10
                global_mape = np.mean(np.abs((all_targets - global_preds) / (np.abs(all_targets) + epsilon))) * 100
                personalized_mape = np.mean(np.abs((all_targets - personalized_preds) / (np.abs(all_targets) + epsilon))) * 100
                
                # Determine which metrics to use based on use_personalized flag
                if use_personalized:
                    client_metrics.append({
                        'loss': avg_personalized_loss,
                        'mae': personalized_mae,
                        'mse': personalized_mse,
                        'mape': personalized_mape,
                        'global_loss': avg_global_loss,
                        'global_mae': global_mae,
                        'global_mse': global_mse,
                        'global_mape': global_mape
                    })
                else:
                    client_metrics.append({
                        'loss': avg_global_loss,
                        'mae': global_mae,
                        'mse': global_mse,
                        'mape': global_mape,
                        'personalized_loss': avg_personalized_loss,
                        'personalized_mae': personalized_mae,
                        'personalized_mse': personalized_mse,
                        'personalized_mape': personalized_mape
                    })
            else:
                client_metrics.append({
                    'loss': float('nan'),
                    'mae': float('nan'),
                    'mse': float('nan'),
                    'mape': float('nan'),
                    'global_loss': float('nan'),
                    'global_mae': float('nan'),
                    'global_mse': float('nan'),
                    'global_mape': float('nan')
                })
        
        return client_metrics
    
    def predict(self, client_idx, data_loader, use_personalized=True):
        """
        Generate predictions for a client.
        Args:
            client_idx: Index of the client
            data_loader: DataLoader for client data
            use_personalized: Whether to use the personalized model
        Returns:
            Dictionary containing predictions and actual targets
        """
        # Choose the appropriate model
        if use_personalized:
            model = self.personalized_models[client_idx]
        else:
            model = self.client_models[client_idx]
        
        model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Unpack batch data
                hf_input, lf_input, lf_target = [b.to(self.device) for b in batch]
                
                # Forward pass
                outputs = model(hf_input, lf_input)
                preds = outputs["lf_pred"]
                
                # Store predictions and targets
                all_preds.append(preds.cpu().numpy())
                all_targets.append(lf_target.cpu().numpy())
        
        # Concatenate predictions and targets
        all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        all_targets = np.concatenate(all_targets, axis=0) if all_targets else np.array([])
        
        return {
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def save_checkpoint(self, path):
        """
        Save model checkpoint.
        Args:
            path: Path to save the checkpoint
        """
        checkpoint = {
            'client_models': [model.state_dict() for model in self.client_models],
            'client_optimizers': [opt.state_dict() for opt in self.client_optimizers],
            'personalized_models': [model.state_dict() for model in self.personalized_models],
            'personalized_optimizers': [opt.state_dict() for opt in self.personalized_optimizers]
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path):
        """
        Load model checkpoint.
        Args:
            path: Path to the checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, model in enumerate(self.client_models):
            model.load_state_dict(checkpoint['client_models'][i])
            
        for i, opt in enumerate(self.client_optimizers):
            opt.load_state_dict(checkpoint['client_optimizers'][i])
            
        for i, model in enumerate(self.personalized_models):
            model.load_state_dict(checkpoint['personalized_models'][i])
            
        for i, opt in enumerate(self.personalized_optimizers):
            opt.load_state_dict(checkpoint['personalized_optimizers'][i])

class SimplePFedAvgSystem(FederatedAVGSystem):
    """A simpler implementation of Personalized Federated Averaging."""
    def __init__(self, client_models, client_optimizers=None, device=None):
        super().__init__(client_models, client_optimizers, device)
        
        # Create personalized models for each client
        self.personalized_models = [copy.deepcopy(model) for model in self.client_models]
        
        # Create optimizers for personalization
        self.personalized_optimizers = [
            optim.Adam(model.parameters(), lr=0.00005)  # Lower learning rate for personalization
            for model in self.personalized_models
        ]
    
    def train_round(self, client_data_loaders, local_epochs=1, client_weights=None):
        """
        Perform one round of federated training with global models.
        Args:
            client_data_loaders: List of DataLoaders for each client
            local_epochs: Number of local epochs for each client
            client_weights: Optional weights for averaging (default: equal weights)
        Returns:
            Dictionary with training metrics
        """
        # Regular FedAvg training
        metrics = super().train_round(client_data_loaders, local_epochs, client_weights)
        
        return metrics
    
    def personalize(self, client_data_loaders, personalization_epochs=5):
        """
        Perform personalization for each client by fine-tuning on local data.
        Args:
            client_data_loaders: List of DataLoaders for each client
            personalization_epochs: Number of epochs for personalization
        Returns:
            Dictionary with personalization metrics
        """
        metrics = {
            'personalized_losses': [0.0] * self.num_clients,
            'batch_count': [0] * self.num_clients
        }
        
        # For each client
        for client_idx, data_loader in enumerate(client_data_loaders):
            # Copy the global model to the personalized model
            self.personalized_models[client_idx].load_state_dict(
                self.client_models[client_idx].state_dict()
            )
            
            # Set to training mode
            self.personalized_models[client_idx].to(self.device)
            self.personalized_models[client_idx].train()
            
            # Fine-tune the personalized model
            personalized_loss = 0.0
            batch_count = 0
            
            # Perform personalization for specified number of epochs
            for epoch in range(personalization_epochs):
                epoch_loss = 0.0
                epoch_batches = 0
                
                for batch in data_loader:
                    # Get batch data
                    hf_input, lf_input, lf_target = [b.to(self.device) for b in batch]
                    
                    # Forward pass
                    self.personalized_optimizers[client_idx].zero_grad()
                    outputs = self.personalized_models[client_idx](hf_input, lf_input)
                    loss = self.criterion(outputs["lf_pred"], lf_target)
                    
                    # Backward and optimize
                    loss.backward()
                    self.personalized_optimizers[client_idx].step()
                    
                    # Track metrics
                    epoch_loss += loss.item()
                    epoch_batches += 1
                
                # Add to overall metrics
                if epoch_batches > 0:
                    personalized_loss += epoch_loss / epoch_batches
                batch_count += 1
            
            # Store average personalization metrics across epochs
            if batch_count > 0:
                metrics['personalized_losses'][client_idx] = personalized_loss / batch_count
            metrics['batch_count'][client_idx] = batch_count
        
        return metrics
    
    def evaluate(self, client_data_loaders, use_personalized=True):
        """
        Evaluate models for all clients.
        Args:
            client_data_loaders: List of DataLoaders for each client
            use_personalized: Whether to use personalized models for evaluation
        Returns:
            Dictionary containing evaluation metrics for each client
        """
        # Set models to evaluation mode
        for model in self.client_models:
            model.eval()
        
        for model in self.personalized_models:
            model.eval()
        
        metrics = {
            'global_losses': [0.0] * self.num_clients,
            'personalized_losses': [0.0] * self.num_clients,
            'batch_count': [0] * self.num_clients,
            'global_preds': [[] for _ in range(self.num_clients)],
            'personalized_preds': [[] for _ in range(self.num_clients)],
            'targets': [[] for _ in range(self.num_clients)]
        }
        
        with torch.no_grad():
            for client_idx, data_loader in enumerate(client_data_loaders):
                for batch in data_loader:
                    # Unpack batch data
                    hf_input, lf_input, lf_target = [b.to(self.device) for b in batch]
                    
                    # Forward pass on global model
                    global_outputs = self.client_models[client_idx](hf_input, lf_input)
                    global_preds = global_outputs["lf_pred"]
                    
                    # Forward pass on personalized model
                    personalized_outputs = self.personalized_models[client_idx](hf_input, lf_input)
                    personalized_preds = personalized_outputs["lf_pred"]
                    
                    # Calculate losses
                    global_loss = self.criterion(global_preds, lf_target)
                    personalized_loss = self.criterion(personalized_preds, lf_target)
                    
                    # Update metrics
                    metrics['global_losses'][client_idx] += global_loss.item()
                    metrics['personalized_losses'][client_idx] += personalized_loss.item()
                    metrics['batch_count'][client_idx] += 1
                    
                    # Store predictions and targets
                    metrics['global_preds'][client_idx].append(global_preds.cpu().numpy())
                    metrics['personalized_preds'][client_idx].append(personalized_preds.cpu().numpy())
                    metrics['targets'][client_idx].append(lf_target.cpu().numpy())
        
        # Calculate final metrics
        client_metrics = []
        
        for client_idx in range(self.num_clients):
            if metrics['batch_count'][client_idx] > 0:
                # Average losses
                avg_global_loss = metrics['global_losses'][client_idx] / metrics['batch_count'][client_idx]
                avg_personalized_loss = metrics['personalized_losses'][client_idx] / metrics['batch_count'][client_idx]
                
                # Concatenate predictions and targets
                all_global_preds = np.concatenate(metrics['global_preds'][client_idx], axis=0)
                all_personalized_preds = np.concatenate(metrics['personalized_preds'][client_idx], axis=0)
                all_targets = np.concatenate(metrics['targets'][client_idx], axis=0)
                
                # Global model metrics
                global_mae = np.mean(np.abs(all_global_preds - all_targets))
                global_mse = np.mean(np.square(all_global_preds - all_targets))
                
                # Personalized model metrics
                personalized_mae = np.mean(np.abs(all_personalized_preds - all_targets))
                personalized_mse = np.mean(np.square(all_personalized_preds - all_targets))
                
                # Calculate MAPE with safeguard against division by zero
                epsilon = 1e-10
                global_mape = np.mean(np.abs((all_targets - all_global_preds) / (np.abs(all_targets) + epsilon))) * 100
                personalized_mape = np.mean(np.abs((all_targets - all_personalized_preds) / (np.abs(all_targets) + epsilon))) * 100
                
                client_metrics.append({
                    'global_loss': avg_global_loss,
                    'global_mae': global_mae,
                    'global_mse': global_mse,
                    'global_mape': global_mape,
                    'personalized_loss': avg_personalized_loss,
                    'personalized_mae': personalized_mae,
                    'personalized_mse': personalized_mse,
                    'personalized_mape': personalized_mape
                })
            else:
                client_metrics.append({
                    'global_loss': float('nan'),
                    'global_mae': float('nan'),
                    'global_mse': float('nan'),
                    'global_mape': float('nan'),
                    'personalized_loss': float('nan'),
                    'personalized_mae': float('nan'),
                    'personalized_mse': float('nan'),
                    'personalized_mape': float('nan')
                })
        
        return client_metrics
    
    def predict(self, client_idx, data_loader, use_personalized=True):
        """
        Generate predictions for a client.
        Args:
            client_idx: Index of the client
            data_loader: DataLoader for client data
            use_personalized: Whether to use the personalized model
        Returns:
            Dictionary containing predictions and actual targets
        """
        # Choose the appropriate model
        if use_personalized:
            model = self.personalized_models[client_idx]
        else:
            model = self.client_models[client_idx]
        
        model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Unpack batch data
                hf_input, lf_input, lf_target = [b.to(self.device) for b in batch]
                
                # Forward pass
                outputs = model(hf_input, lf_input)
                preds = outputs["lf_pred"]
                
                # Store predictions and targets
                all_preds.append(preds.cpu().numpy())
                all_targets.append(lf_target.cpu().numpy())
        
        # Concatenate predictions and targets
        all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        all_targets = np.concatenate(all_targets, axis=0) if all_targets else np.array([])
        
        return {
            'predictions': all_preds,
            'targets': all_targets
        }

class PrivacyPreservingFedProxSystem:
    """Privacy-preserving federated learning system with secure aggregation and FedProx."""
    def __init__(self, client_models, client_optimizers=None, 
                 noise_scale=0.1, clip_norm=1.0, fedprox_mu=0.01,
                 enable_secure_agg=True, device=None):
        self.client_models = client_models
        self.num_clients = len(client_models)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Privacy parameters
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
        self.enable_secure_agg = enable_secure_agg
        
        # FedProx parameter
        self.fedprox_mu = fedprox_mu
        
        # Move models to device
        for model in self.client_models:
            model.to(self.device)
        
        # Create proximal term calculators for each client
        self.proximal_calculators = [
            ProximalTerm(mu=fedprox_mu) for _ in range(self.num_clients)
        ]
        
        # Set default optimizers if not provided
        if client_optimizers is None:
            self.base_optimizers = [
                optim.Adam(model.parameters(), lr=0.001)
                for model in self.client_models
            ]
        else:
            self.base_optimizers = client_optimizers
        
        # Wrap optimizers with DP and proximal term
        self.client_optimizers = [
            DPOptimizerWithProximal(
                optimizer=optimizer, 
                proximal_calculator=self.proximal_calculators[i],
                noise_scale=noise_scale, 
                max_grad_norm=clip_norm
            )
            for i, optimizer in enumerate(self.base_optimizers)
        ]
        
        # Initialize secure aggregation helper
        self.secure_agg = SecureAggregation(device=self.device)
        
        # Create personalized models for each client
        self.personalized_models = [copy.deepcopy(model) for model in self.client_models]
        
        # Create optimizers for personalization (with stronger privacy)
        self.personalized_optimizers = [
            DPOptimizer(
                optim.Adam(model.parameters(), lr=0.0001),  # Lower learning rate for personalization
                noise_scale=0,  # No additional noise for personalization
                max_grad_norm=0  # No clipping for personalization
            )
            for model in self.personalized_models
        ]
            
        self.criterion = nn.MSELoss()
    
    def train_round(self, client_data_loaders, local_epochs=1, client_weights=None):
        """
        Perform one round of private federated training with FedProx.
        """
        # Set models to training mode
        for model in self.client_models:
            model.train()
        
        # Track metrics
        metrics = {
            'losses': [0.0] * self.num_clients,
            'proximal_terms': [0.0] * self.num_clients,
            'batch_count': [0] * self.num_clients
        }
        
        # Default to equal weights if not provided
        if client_weights is None:
            client_weights = [1.0 / self.num_clients] * self.num_clients
        
        # Store initial model state for computing updates
        initial_state = copy.deepcopy(self.client_models[0].state_dict())
        
        # Update proximal term calculators with initial state
        for calculator in self.proximal_calculators:
            calculator.set_global_params({k: v.clone() for k, v in initial_state.items()})
        
        # Step 1: Local client training
        client_updates = []
        
        for client_idx, data_loader in enumerate(client_data_loaders):
            # Reset model to global state
            self.client_models[client_idx].load_state_dict(initial_state)
            
            # Local training
            for epoch in range(local_epochs):
                print("In Epoch: ", epoch)
                epoch_loss = 0.0
                epoch_prox_term = 0.0
                batch_count = 0
                
                for batch in data_loader:
                    # Get local batch
                    hf_input, lf_input, lf_target = [b.to(self.device) for b in batch]
                    
                    # Forward pass on local model
                    self.client_optimizers[client_idx].zero_grad()
                    outputs = self.client_models[client_idx](hf_input, lf_input)
                    
                    # Prediction loss
                    loss = self.criterion(outputs["lf_pred"], lf_target)
                    
                    # Add proximal term (FedProx) - L2 penalty between local and global model
                    proximal_term = self.proximal_calculators[client_idx].compute_proximal_term(self.client_models[client_idx])
                    
                    # Combined loss with proximal term
                    total_loss = loss + proximal_term
                    
                    # Backward
                    total_loss.backward()
                    
                    # Step with DP optimizer (handles clipping and noise)
                    self.client_optimizers[client_idx].step()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    epoch_prox_term += proximal_term.item() if isinstance(proximal_term, torch.Tensor) else proximal_term
                    batch_count += 1
                
                # Add the local epoch metrics
                if batch_count > 0:
                    metrics['losses'][client_idx] += epoch_loss / batch_count
                    metrics['proximal_terms'][client_idx] += epoch_prox_term / batch_count
                metrics['batch_count'][client_idx] += 1
        
            # Calculate average losses per client
            if metrics['batch_count'][client_idx] > 0:
                metrics['losses'][client_idx] /= metrics['batch_count'][client_idx]
                metrics['proximal_terms'][client_idx] /= metrics['batch_count'][client_idx]
            
            # Compute model update
            updated_state = self.client_models[client_idx].state_dict()
            client_update = {}
            
            for key in updated_state.keys():
                client_update[key] = updated_state[key] - initial_state[key]
            
            client_updates.append(client_update)
        
        # Step 2: Secure aggregation of updates
        if self.enable_secure_agg:
            # Get model shape dictionary
            model_shape_dict = self.secure_agg.compute_model_shape_dict(self.client_models[0])
            
            # Generate masking vectors
            masks = self.secure_agg.generate_masking_vectors(model_shape_dict, self.num_clients)
            
            # Apply masks (in a real system, clients would only share the masked updates)
            masked_updates = [
                self.secure_agg.apply_mask(update, mask)
                for update, mask in zip(client_updates, masks)
            ]
            
            # Server aggregates masked updates (masks sum to zero, so they cancel out)
            aggregated_update = {}
            for key in model_shape_dict:
                aggregated_update[key] = sum(
                    masked_updates[i][key] * client_weights[i] for i in range(self.num_clients)
                )
        else:
            # Standard weighted aggregation (non-secure)
            aggregated_update = {}
            for key in client_updates[0].keys():
                aggregated_update[key] = sum(
                    client_updates[i][key] * client_weights[i] for i in range(self.num_clients)
                )
        
        # Apply aggregated update to all client models
        for model in self.client_models:
            updated_global_state = copy.deepcopy(initial_state)
            for key in updated_global_state:
                updated_global_state[key] += aggregated_update[key]
            model.load_state_dict(updated_global_state)
        
        return metrics
    
    def personalize(self, client_data_loaders, personalization_epochs=3):
        """
        Perform personalization for each client with privacy guarantees.
        """
        metrics = {
            'personalized_losses': [0.0] * self.num_clients,
            'batch_count': [0] * self.num_clients
        }
        
        # For each client
        for client_idx, data_loader in enumerate(client_data_loaders):
            # Copy the global model to the personalized model
            self.personalized_models[client_idx].load_state_dict(
                self.client_models[client_idx].state_dict()
            )
            
            # Set to training mode
            self.personalized_models[client_idx].to(self.device)
            self.personalized_models[client_idx].train()
            
            # Fine-tune with stronger DP
            personalized_loss = 0.0
            batch_count = 0
            
            # Personalization
            for epoch in range(personalization_epochs):
                epoch_loss = 0.0
                epoch_batches = 0
                
                for batch in data_loader:
                    # Get batch data
                    hf_input, lf_input, lf_target = [b.to(self.device) for b in batch]
                    
                    # Forward pass
                    self.personalized_optimizers[client_idx].zero_grad()
                    outputs = self.personalized_models[client_idx](hf_input, lf_input)
                    loss = self.criterion(outputs["lf_pred"], lf_target)
                    
                    # Backward
                    loss.backward()
                    
                    # Step with personalized DP optimizer
                    self.personalized_optimizers[client_idx].step()
                    
                    # Track metrics
                    epoch_loss += loss.item()
                    epoch_batches += 1
                
                # Add to overall metrics
                if epoch_batches > 0:
                    personalized_loss += epoch_loss / epoch_batches
                batch_count += 1
            
            # Store average personalization metrics across epochs
            if batch_count > 0:
                metrics['personalized_losses'][client_idx] = personalized_loss / batch_count
            metrics['batch_count'][client_idx] = batch_count
        
        return metrics
    
    def evaluate(self, client_data_loaders, use_personalized=True):
        """
        Evaluate models for all clients.
        """
        # Choose which models to evaluate
        if use_personalized:
            eval_models = self.personalized_models
        else:
            eval_models = self.client_models
        
        # Set models to evaluation mode
        for model in eval_models:
            model.eval()
        
        metrics = {
            'losses': [0.0] * self.num_clients,
            'batch_count': [0] * self.num_clients,
            'all_preds': [[] for _ in range(self.num_clients)],
            'all_targets': [[] for _ in range(self.num_clients)]
        }
        
        with torch.no_grad():
            for client_idx, data_loader in enumerate(client_data_loaders):
                for batch in data_loader:
                    # Unpack batch data
                    hf_input, lf_input, lf_target = [b.to(self.device) for b in batch]
                    
                    # Forward pass without adding noise (evaluation mode)
                    outputs = eval_models[client_idx](hf_input, lf_input)
                    preds = outputs["lf_pred"]
                    
                    # Calculate loss
                    loss = self.criterion(preds, lf_target)
                    
                    # Update metrics
                    metrics['losses'][client_idx] += loss.item()
                    metrics['batch_count'][client_idx] += 1
                    
                    # Store predictions and targets
                    metrics['all_preds'][client_idx].append(preds.cpu().numpy())
                    metrics['all_targets'][client_idx].append(lf_target.cpu().numpy())
        
        # Calculate client metrics
        client_metrics = []
        
        for client_idx in range(self.num_clients):
            if metrics['batch_count'][client_idx] > 0:
                # Average loss
                avg_loss = metrics['losses'][client_idx] / metrics['batch_count'][client_idx]
                
                # Concatenate predictions and targets
                all_preds = np.concatenate(metrics['all_preds'][client_idx], axis=0)
                all_targets = np.concatenate(metrics['all_targets'][client_idx], axis=0)
                
                # Calculate additional metrics
                mae = np.mean(np.abs(all_preds - all_targets))
                mse = np.mean(np.square(all_preds - all_targets))
                
                # Calculate MAPE with safeguard against division by zero
                epsilon = 1e-10
                mape = np.mean(np.abs((all_targets - all_preds) / (np.abs(all_targets) + epsilon))) * 100
                
                client_metrics.append({
                    'loss': avg_loss,
                    'mae': mae,
                    'mse': mse,
                    'mape': mape
                })
            else:
                client_metrics.append({
                    'loss': float('nan'),
                    'mae': float('nan'),
                    'mse': float('nan'),
                    'mape': float('nan')
                })
        
        return client_metrics

#####################################################
# Main Program
#####################################################
# TODO: Move to Trainer Folder.
@DeprecationWarning
def train_personalized_fedavg_system(
    client_datasets, hidden_dim=64, num_layers=2, dropout=0.2,
    batch_size=32, learning_rate=0.001, meta_learning_rate=0.01,
    epochs=30, local_epochs=1, personalization_steps=5, device=None
):
    """
    Train the federated system using Personalized FedAVG.
    Args:
        client_datasets: Dictionary of client datasets
        hidden_dim: Hidden dimension for models
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        batch_size: Batch size for training
        learning_rate: Learning rate for client models
        meta_learning_rate: Learning rate for meta-update
        epochs: Number of communication rounds
        local_epochs: Number of local epochs per round
        personalization_steps: Number of steps for personalization
        device: Device to use for computation
    Returns:
        Trained system and training history
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loaders = []
    test_loaders = []
    
    for client_id, client_data in client_datasets.items():
        train_loader = DataLoader(
            client_data['train_dataset'], 
            batch_size=batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            client_data['test_dataset'], 
            batch_size=batch_size
        )
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    # Get input dimensions (assuming all clients have same dimensions)
    first_client = next(iter(client_datasets.values()))
    hf_input_dim = first_client['hf_data_shape'][1]
    lf_input_dim = first_client['lf_data_shape'][1]
    lf_output_dim = first_client['lf_data_shape'][1]
    
    # Create client models
    print("Creating client models...")
    client_models = []
    
    for client_id in client_datasets.keys():
        model = FedAVGModel(
            hf_input_dim=hf_input_dim,
            lf_input_dim=lf_input_dim,
            lf_output_dim=lf_output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        client_models.append(model)
    
    # Create optimizers
    client_optimizers = [
        optim.Adam(model.parameters(), lr=learning_rate) 
        for model in client_models
    ]
    
    # Create personalized federated system
    print("Initializing personalized FedAVG system...")
    system = PersonalizedFedAVGSystem(
        client_models=client_models,
        client_optimizers=client_optimizers,
        meta_learning_rate=meta_learning_rate,
        device=device
    )
    
    # Training loop
    print(f"Starting training for {epochs} rounds with {local_epochs} local epochs each...")
    history = {
        'train_metrics': [],
        'val_metrics': [],
        'meta_metrics': [],
        'best_round': 0,
        'best_val_loss': float('inf'),
        'best_personalized_val_loss': float('inf')
    }
    
    best_model_state = None
    start_time = time.time()
    
    # Calculate client weights based on dataset sizes
    client_data_sizes = [len(client_datasets[client_id]['train_dataset']) for client_id in client_datasets]
    total_data_size = sum(client_data_sizes)
    client_weights = [size / total_data_size for size in client_data_sizes]
    
    print(f"Client weights for averaging: {client_weights}")
    
    for round_idx in range(epochs):
        round_start_time = time.time()
        
        # Meta-update step (MAML-style personalization)
        meta_metrics = system.meta_update(
            client_data_loaders=train_loaders,
            meta_batch_size=2  # Number of batches for meta-learning
        )
        
        # Train one round with personalization
        train_metrics = system.train_round(
            client_data_loaders=train_loaders,
            local_epochs=local_epochs,
            client_weights=client_weights,
            personalization_steps=personalization_steps
        )
        
        # Evaluate with both global and personalized models
        val_metrics = system.evaluate(
            test_loaders, 
            use_personalized=True  # Use personalized models for primary metrics
        )
        
        # Calculate average validation losses
        avg_global_val_loss = np.mean([m['global_loss'] for m in val_metrics])
        avg_personalized_val_loss = np.mean([m['loss'] for m in val_metrics])
        
        # Check for improvement
        if avg_personalized_val_loss < history['best_personalized_val_loss']:
            history['best_personalized_val_loss'] = avg_personalized_val_loss
            history['best_round'] = round_idx
            best_model_state = copy.deepcopy({
                'client_models': [model.state_dict() for model in system.client_models],
                'personalized_models': [model.state_dict() for model in system.personalized_models]
            })
        
        # Save metrics
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        history['meta_metrics'].append(meta_metrics)
        
        # Print progress
        round_time = time.time() - round_start_time
        print(f"Round {round_idx+1}/{epochs} - Time: {round_time:.2f}s")
        
        # Print client losses
        for client_idx in range(len(client_models)):
            print(f"  Client {client_idx} - Train Loss: {train_metrics['losses'][client_idx]:.4f}, "
                  f"Personalized Train Loss: {train_metrics.get('personalized_losses', [0]*len(client_models))[client_idx]:.4f}")
            print(f"    Global Val Loss: {val_metrics[client_idx]['global_loss']:.4f}, "
                  f"Global Val MAPE: {val_metrics[client_idx]['global_mape']:.2f}%")
            print(f"    Personalized Val Loss: {val_metrics[client_idx]['loss']:.4f}, "
                  f"Personalized Val MAPE: {val_metrics[client_idx]['mape']:.2f}%")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Best personalized validation loss {history['best_personalized_val_loss']:.4f} at round {history['best_round']+1}")
    
    # Restore best model if available
    if best_model_state:
        for i, model in enumerate(system.client_models):
            model.load_state_dict(best_model_state['client_models'][i])
        for i, model in enumerate(system.personalized_models):
            model.load_state_dict(best_model_state['personalized_models'][i])
    
    return system, history

def train_simple_pfedavg_system(
    client_datasets, hidden_dim=64, num_layers=2, dropout=0.2,
    batch_size=32, learning_rate=0.001, personalization_lr=0.00005,
    rounds=10, local_epochs=5, personalization_epochs=3, device=None
):
    """
    Train a simple personalized federated learning system.
    Args:
        client_datasets: Dictionary of client datasets
        hidden_dim: Hidden dimension for models
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        batch_size: Batch size for training
        learning_rate: Learning rate for client models
        personalization_lr: Learning rate for personalization
        rounds: Number of communication rounds
        local_epochs: Number of local epochs per round
        personalization_epochs: Number of epochs for personalization
        device: Device to use for computation
    Returns:
        Trained system and training history
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loaders = []
    test_loaders = []
    
    for client_id, client_data in client_datasets.items():
        train_loader = DataLoader(
            client_data['train_dataset'], 
            batch_size=batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            client_data['test_dataset'], 
            batch_size=batch_size
        )
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    # Get input dimensions (assuming all clients have same dimensions)
    first_client = next(iter(client_datasets.values()))
    hf_input_dim = first_client['hf_data_shape'][1]
    lf_input_dim = first_client['lf_data_shape'][1]
    lf_output_dim = first_client['lf_data_shape'][1]
    
    # Create client models
    print("Creating client models...")
    client_models = []
    
    for client_id in client_datasets.keys():
        model = FedAVGModel(
            hf_input_dim=hf_input_dim,
            lf_input_dim=lf_input_dim,
            lf_output_dim=lf_output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        client_models.append(model)
    
    # Create optimizers
    client_optimizers = [
        optim.Adam(model.parameters(), lr=learning_rate) 
        for model in client_models
    ]
    
    # Create personalized federated system
    print("Initializing simple personalized FedAvg system...")
    system = SimplePFedAvgSystem(
        client_models=client_models,
        client_optimizers=client_optimizers,
        device=device
    )
    
    # Set personalization learning rate
    for opt in system.personalized_optimizers:
        for param_group in opt.param_groups:
            param_group['lr'] = personalization_lr
    
    # Training loop
    print(f"Starting training for {rounds} rounds with {local_epochs} local epochs each...")
    history = {
        'train_metrics': [],
        'personalization_metrics': [],
        'val_metrics': [],
        'best_round': 0,
        'best_global_val_loss': float('inf'),
        'best_personalized_val_loss': float('inf')
    }
    
    best_model_state = None
    start_time = time.time()
    
    # Calculate client weights based on dataset sizes
    client_data_sizes = [len(client_datasets[client_id]['train_dataset']) for client_id in client_datasets]
    total_data_size = sum(client_data_sizes)
    client_weights = [size / total_data_size for size in client_data_sizes]
    
    print(f"Client weights for averaging: {client_weights}")
    
    for round_idx in range(rounds):
        round_start_time = time.time()
        
        # Step 1: Standard FedAvg training
        train_metrics = system.train_round(
            client_data_loaders=train_loaders,
            local_epochs=local_epochs,
            client_weights=client_weights
        )
        
        # Step 2: Personalization through local fine-tuning
        personalization_metrics = system.personalize(
            client_data_loaders=train_loaders,
            personalization_epochs=personalization_epochs
        )
        
        # Step 3: Evaluate both global and personalized models
        val_metrics = system.evaluate(test_loaders)
        
        # Calculate average validation losses
        avg_global_val_loss = np.mean([m['global_loss'] for m in val_metrics])
        avg_personalized_val_loss = np.mean([m['personalized_loss'] for m in val_metrics])
        
        # Check for improvement
        if avg_personalized_val_loss < history['best_personalized_val_loss']:
            history['best_personalized_val_loss'] = avg_personalized_val_loss
            history['best_round'] = round_idx
            best_model_state = copy.deepcopy({
                'client_models': [model.state_dict() for model in system.client_models],
                'personalized_models': [model.state_dict() for model in system.personalized_models]
            })
        
        # Save metrics
        history['train_metrics'].append(train_metrics)
        history['personalization_metrics'].append(personalization_metrics)
        history['val_metrics'].append(val_metrics)
        
        # Print progress
        round_time = time.time() - round_start_time
        print(f"Round {round_idx+1}/{rounds} - Time: {round_time:.2f}s")
        
        # Print client losses
        for client_idx in range(len(client_models)):
            global_train_loss = train_metrics['losses'][client_idx]
            personalized_train_loss = personalization_metrics['personalized_losses'][client_idx]
            
            print(f"  Client {client_idx} - Global Train Loss: {global_train_loss:.4f}, "
                  f"Personalized Train Loss: {personalized_train_loss:.4f}")
            
            global_val_loss = val_metrics[client_idx]['global_loss']
            global_val_mape = val_metrics[client_idx]['global_mape']
            personalized_val_loss = val_metrics[client_idx]['personalized_loss']
            personalized_val_mape = val_metrics[client_idx]['personalized_mape']
            
            print(f"    Global Val Loss: {global_val_loss:.4f}, "
                  f"Global Val MAPE: {global_val_mape:.2f}%")
            print(f"    Personalized Val Loss: {personalized_val_loss:.4f}, "
                  f"Personalized Val MAPE: {personalized_val_mape:.2f}%")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Best personalized validation loss {history['best_personalized_val_loss']:.4f} at round {history['best_round']+1}")
    
    # Restore best model if available
    if best_model_state:
        for i, model in enumerate(system.client_models):
            model.load_state_dict(best_model_state['client_models'][i])
        for i, model in enumerate(system.personalized_models):
            model.load_state_dict(best_model_state['personalized_models'][i])
    
    return system, history

def train_private_fedprox_system(
    client_datasets, hidden_dim=64, num_layers=2, dropout=0.2,
    batch_size=32, learning_rate=0.001, personalization_lr=0.0001,
    rounds=10, local_epochs=5, personalization_epochs=3, 
    noise_scale=0.1, clip_norm=1.0, encoder_noise_scale=0.05,
    fedprox_mu=0.01, enable_secure_agg=True, device=None
):
    """
    Train a privacy-preserving federated learning system with FedProx.
    Args:
        client_datasets: Dictionary of client datasets
        hidden_dim: Hidden dimension for models
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        batch_size: Batch size for training
        learning_rate: Learning rate for global models
        personalization_lr: Learning rate for personalization
        rounds: Number of communication rounds
        local_epochs: Number of local epochs per round
        personalization_epochs: Number of epochs for personalization
        noise_scale: Noise scale for DP-SGD
        clip_norm: Gradient clipping norm for DP-SGD
        encoder_noise_scale: Noise scale for encoder outputs
        fedprox_mu: FedProx regularization coefficient
        enable_secure_agg: Whether to use secure aggregation
        device: Device to use for computation
    Returns:
        Trained system and training history
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loaders = []
    test_loaders = []
    
    for client_id, client_data in client_datasets.items():
        train_loader = DataLoader(
            client_data['train_dataset'], 
            batch_size=batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            client_data['test_dataset'], 
            batch_size=batch_size
        )
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    # Get input dimensions (assuming all clients have same dimensions)
    first_client = next(iter(client_datasets.values()))
    hf_input_dim = first_client['hf_data_shape'][1]
    lf_input_dim = first_client['lf_data_shape'][1]
    lf_output_dim = first_client['lf_data_shape'][1]
    
    # Create privacy-preserving client models
    print("Creating privacy-preserving client models...")
    client_models = []
    
    for client_id in client_datasets.keys():
        model = PrivacyPreservingFedAVGModel(
            hf_input_dim=hf_input_dim,
            lf_input_dim=lf_input_dim,
            lf_output_dim=lf_output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            noise_scale=encoder_noise_scale,
            clip_bound=1.0
        )
        client_models.append(model)
    
    # Create optimizers
    client_optimizers = [
        optim.Adam(model.parameters(), lr=learning_rate) 
        for model in client_models
    ]
    
    # Create privacy-preserving federated system with FedProx
    print("Initializing privacy-preserving FedProx federated system...")
    system = PrivacyPreservingFedProxSystem(
        client_models=client_models,
        client_optimizers=client_optimizers,
        noise_scale=noise_scale,
        clip_norm=clip_norm,
        fedprox_mu=fedprox_mu,  # FedProx regularization strength
        enable_secure_agg=enable_secure_agg,
        device=device
    )
    
    # For personalized optimizers, set custom learning rate
    for opt in system.personalized_optimizers:
        for param_group in opt.optimizer.param_groups:
            param_group['lr'] = personalization_lr
    
    # Training loop
    print(f"Starting private FedProx training for {rounds} rounds with {local_epochs} local epochs each...")
    history = {
        'train_metrics': [],
        'personalization_metrics': [],
        'val_metrics': [],
        'best_round': 0,
        'best_global_val_loss': float('inf'),
        'best_personalized_val_loss': float('inf')
    }
    
    best_model_state = None
    start_time = time.time()
    
    # Calculate client weights based on dataset sizes
    client_data_sizes = [len(client_datasets[client_id]['train_dataset']) for client_id in client_datasets]
    total_data_size = sum(client_data_sizes)
    client_weights = [size / total_data_size for size in client_data_sizes]
    
    print(f"Client weights for averaging: {client_weights}")
    
    for round_idx in range(rounds):
        round_start_time = time.time()
        
        # Step 1: Privacy-preserving FedProx training
        train_metrics = system.train_round(
            client_data_loaders=train_loaders,
            local_epochs=local_epochs,
            client_weights=client_weights
        )
        
        # Step 2: Personalization through local fine-tuning with DP
        personalization_metrics = system.personalize(
            client_data_loaders=train_loaders,
            personalization_epochs=personalization_epochs
        )
        
        # Step 3: Evaluate both global and personalized models
        global_val_metrics = system.evaluate(test_loaders, use_personalized=False)
        personalized_val_metrics = system.evaluate(test_loaders, use_personalized=True)
        
        # Calculate average validation losses
        avg_global_val_loss = np.mean([m['loss'] for m in global_val_metrics])
        avg_personalized_val_loss = np.mean([m['loss'] for m in personalized_val_metrics])
        
        # Check for improvement
        if avg_personalized_val_loss < history['best_personalized_val_loss']:
            history['best_personalized_val_loss'] = avg_personalized_val_loss
            history['best_round'] = round_idx
            best_model_state = {
                'client_models': [model.state_dict() for model in system.client_models],
                'personalized_models': [model.state_dict() for model in system.personalized_models]
            }
        
        # Save metrics
        history['train_metrics'].append(train_metrics)
        history['personalization_metrics'].append(personalization_metrics)
        history['val_metrics'].append({
            'global': global_val_metrics,
            'personalized': personalized_val_metrics
        })
        
        # Print progress
        round_time = time.time() - round_start_time
        print(f"Round {round_idx+1}/{rounds} - Time: {round_time:.2f}s")
        
        # Print client losses
        for client_idx in range(len(client_models)):
            global_train_loss = train_metrics['losses'][client_idx]
            proximal_term = train_metrics['proximal_terms'][client_idx]
            personalized_train_loss = personalization_metrics['personalized_losses'][client_idx]
            
            print(f"  Client {client_idx} - Global Train Loss: {global_train_loss:.4f}, "
                  f"Proximal Term: {proximal_term:.4f}, "
                  f"Personalized Train Loss: {personalized_train_loss:.4f}")
            
            global_val_loss = global_val_metrics[client_idx]['loss']
            global_val_mape = global_val_metrics[client_idx]['mape']
            personalized_val_loss = personalized_val_metrics[client_idx]['loss']
            personalized_val_mape = personalized_val_metrics[client_idx]['mape']
            
            print(f"    Global Val Loss: {global_val_loss:.4f}, "
                  f"Global Val MAPE: {global_val_mape:.2f}%")
            print(f"    Personalized Val Loss: {personalized_val_loss:.4f}, "
                  f"Personalized Val MAPE: {personalized_val_mape:.2f}%")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Best personalized validation loss {history['best_personalized_val_loss']:.4f} at round {history['best_round']+1}")
    
    # Restore best model if available
    if best_model_state:
        for i, model in enumerate(system.client_models):
            model.load_state_dict(best_model_state['client_models'][i])
        for i, model in enumerate(system.personalized_models):
            model.load_state_dict(best_model_state['personalized_models'][i])
    
    return system, history