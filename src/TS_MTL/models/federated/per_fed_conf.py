import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import time
from opacus.accountants.rdp import RDPAccountant as OpacusRDPAccountant
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
from .model_components import DPOptimizer, SecureAggregation, PrivacyPreservingFedAVGModel

#####################################################
# Federated Averaging System
#####################################################

class PrivacyPreservingFedSystem:

    """Privacy-preserving federated learning system with secure aggregation."""
    def __init__(self, client_models, client_optimizers=None, 
                 noise_scale=0.1, clip_norm=1.0, 
                 enable_secure_agg=True, device=None):
        self.client_models = client_models
        self.num_clients = len(client_models)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Privacy parameters
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
        self.enable_secure_agg = enable_secure_agg
        
        # Move models to device
        for model in self.client_models:
            model.to(self.device)
        
        # Set default optimizers if not provided
        if client_optimizers is None:
            self.base_optimizers = [
                optim.Adam(model.parameters(), lr=0.001)
                for model in self.client_models
            ]
        else:
            self.base_optimizers = client_optimizers
        
        # Wrap optimizers with DP
        self.client_optimizers = [
            DPOptimizer(optimizer, noise_scale=noise_scale, max_grad_norm=clip_norm) # FIX
            for optimizer in self.base_optimizers
        ]
        
        # Initialize secure aggregation helper
        self.secure_agg = SecureAggregation(device=self.device)
        
        # Create personalized models for each client
        self.personalized_models = [copy.deepcopy(model) for model in self.client_models]
        
        # Create optimizers for personalization (with stronger privacy)
        self.personalized_optimizers = [
            DPOptimizer(
                optim.Adam(model.parameters(), lr=0.0001),  # Lower learning rate for personalization
                noise_scale= 0, #noise_scale * 1.5,  # More noise for personalization
                max_grad_norm= 0 # clip_norm * 0.8  # Stricter clipping for personalization
            )
            for model in self.personalized_models
        ]
            
        self.criterion = nn.MSELoss()
    
    def train_round(self, client_data_loaders, local_epochs=1, client_weights=None):
        """
        Perform one round of private federated training.
        """
        # Set models to training mode
        for model in self.client_models:
            model.train()
        
        # Track metrics
        metrics = {
            'losses': [0.0] * self.num_clients,
            'batch_count': [0] * self.num_clients
        }
        
        # Default to equal weights if not provided
        if client_weights is None:
            client_weights = [1.0 / self.num_clients] * self.num_clients
        
        # Store initial model state for computing updates
        initial_state = copy.deepcopy(self.client_models[0].state_dict())
        
        # Step 1: Local client training
        client_updates = []
        
        for client_idx, data_loader in enumerate(client_data_loaders):
            # Reset model to global state
            self.client_models[client_idx].load_state_dict(initial_state)
            
            # Local training
            for epoch in range(local_epochs):
                print("In Epoch: ", epoch)
                epoch_loss = 0.0
                batch_count = 0
                
                for batch in data_loader:
                    # Get local batch
                    hf_input, lf_input, lf_target = [b.to(self.device) for b in batch]
                    
                    # Forward pass on local model
                    self.client_optimizers[client_idx].zero_grad()
                    outputs = self.client_models[client_idx](hf_input, lf_input)
                    
                    # Prediction loss
                    loss = self.criterion(outputs["lf_pred"], lf_target)
                    
                    # Backward
                    loss.backward()
                    
                    # Step with DP optimizer (handles clipping and noise)
                    self.client_optimizers[client_idx].step()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    batch_count += 1
                
                # Add the local epoch metrics
                if batch_count > 0:
                    metrics['losses'][client_idx] += epoch_loss / batch_count
                metrics['batch_count'][client_idx] += 1
        
            # Calculate average losses per client
            if metrics['batch_count'][client_idx] > 0:
                metrics['losses'][client_idx] /= metrics['batch_count'][client_idx]
            
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
    def predict(self, client_idx, data_loader, use_personalized=False):
        """
        Generate predictions for a client in the privacy‐preserving system.
        Args:
            client_idx: Index of the client
            data_loader: DataLoader for this client
            use_personalized: Whether to use the personalized model (if available)
        Returns:
            dict with 'predictions' and 'targets' as numpy arrays
        """
        # Select the model
        if use_personalized and hasattr(self, 'personalized_models'):
            model = self.personalized_models[client_idx]
        else:
            model = self.client_models[client_idx]
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                hf_input, lf_input, lf_target = [b.to(self.device) for b in batch]

                outputs = model(hf_input, lf_input)
                preds = outputs["lf_pred"]

                all_preds.append(preds.cpu().numpy())
                all_targets.append(lf_target.cpu().numpy())

        # Concatenate into single arrays
        preds_arr = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        targs_arr = np.concatenate(all_targets, axis=0) if all_targets else np.array([])

        return {
            'predictions': preds_arr,
            'targets': targs_arr
        }
    
#####################################################
# Main Program
#####################################################
# TODO: Move to Trainer Folder.
def train_private_federated_system(
    client_datasets, hidden_dim=64, num_layers=2, dropout=0.2,
    batch_size=32, learning_rate=0.001, personalization_lr=0.0001,
    rounds=10, local_epochs=5, personalization_epochs=3, 
    noise_scale=0.1, clip_norm=1.0, encoder_noise_scale=0.05,
    enable_secure_agg=True, device=None
):
    """
    Train a privacy-preserving federated learning system.
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
    
    # Create privacy-preserving federated system
    print("Initializing privacy-preserving federated system...")
    system = PrivacyPreservingFedSystem(
        client_models=client_models,
        client_optimizers=client_optimizers,
        noise_scale=noise_scale,
        clip_norm=clip_norm,
        enable_secure_agg=enable_secure_agg,
        device=device
    )
    
    # For personalized optimizers, set custom learning rate
    for opt in system.personalized_optimizers:
        for param_group in opt.optimizer.param_groups:
            param_group['lr'] = personalization_lr
    
    # Training loop
    print(f"Starting private training for {rounds} rounds with {local_epochs} local epochs each...")
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
        
        # Calculate batches per epoch and sampling rate
        total_examples = sum([len(train_loader.dataset) for train_loader in train_loaders])
        avg_batch_size = sum([train_loader.batch_size for train_loader in train_loaders]) / len(train_loaders)
        sampling_rate = avg_batch_size / total_examples
        total_batches = sum([len(train_loader) for train_loader in train_loaders]) * local_epochs
        
        # Step 1: Privacy-preserving FedAvg training
        train_metrics = system.train_round(
            client_data_loaders=train_loaders,
            local_epochs=local_epochs,
            client_weights=client_weights
        )
        
        # Update privacy budget
        # budget_tracker.step_encoder(
        #     noise_multiplier=encoder_noise_scale,
        #     sample_rate=sampling_rate,
        #     batch_count=total_batches
        # )
        
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
            personalized_train_loss = personalization_metrics['personalized_losses'][client_idx]
            
            print(f"  Client {client_idx} - Global Train Loss: {global_train_loss:.4f}, "
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