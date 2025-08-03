import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

#####################################################
# Training and Evaluation
#####################################################

class HardParameterSharingTrainer:
    """Trainer for hard parameter sharing model."""
    def __init__(self, model, learning_rate=0.001, device=None):
        self.model = model
        self.learning_rate = learning_rate
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def train_step(self, batch):
        """Perform a single training step."""
        # Unpack batch data
        if len(batch) == 4:  # Includes client_id
            hf_input, lf_input, lf_target, client_id = [
                b.to(self.device) for b in batch
            ]
        else:  # No client_id
            hf_input, lf_input, lf_target = [
                b.to(self.device) for b in batch
            ]
            client_id = None
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(hf_input, lf_input, client_id)
        
        # Calculate loss
        loss = self.criterion(outputs["lf_pred"], lf_target)
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()
        
        return {"loss": loss.item()}
    
    def fit(self, train_loader, val_loader=None, epochs=50, early_stopping_patience=10, verbose=True):
        """
        Train the model.
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: Number of training epochs
            early_stopping_patience: Number of epochs to wait for improvement
            verbose: Whether to print progress
        Returns:
            Dictionary containing training history
        """
        history = {
            "train_loss": [],
            "val_loss": [] if val_loader else None
        }
        
        best_val_loss = float('inf')
        no_improvement_count = 0
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                step_result = self.train_step(batch)
                train_losses.append(step_result["loss"])
                
                # Print progress every 100 batches
                if verbose and batch_idx % 100 == 0 and batch_idx > 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {np.mean(train_losses[-100:]):.4f}")
            
            # Calculate average training loss
            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)
            
            # Validation (if provided)
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics["loss"]
                history["val_loss"].append(val_loss)
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement_count = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        # Restore best model
                        self.model.load_state_dict(best_model_state)
                        break
            
            # Print epoch summary
            if verbose:
                elapsed_time = time.time() - start_time
                log_msg = (f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                           f"Time: {elapsed_time:.2f}s")
                if val_loader:
                    log_msg += f", Val Loss: {val_loss:.4f}"
                print(log_msg)
        
        # Final message
        if verbose:
            total_time = time.time() - start_time
            print(f"Training completed in {total_time:.2f} seconds")
            
        return history
    
    def evaluate(self, data_loader):
        """
        Evaluate the model.
        Args:
            data_loader: DataLoader for evaluation data
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Unpack batch data
                if len(batch) == 4:  # Includes client_id
                    hf_input, lf_input, lf_target, client_id = [
                        b.to(self.device) for b in batch
                    ]
                else:  # No client_id
                    hf_input, lf_input, lf_target = [
                        b.to(self.device) for b in batch
                    ]
                    client_id = None
                
                # Forward pass
                outputs = self.model(hf_input, lf_input, client_id)
                preds = outputs["lf_pred"]
                
                # Calculate loss
                loss = self.criterion(preds, lf_target)
                losses.append(loss.item())
                
                # Store predictions and targets for additional metrics
                all_preds.append(preds.cpu().numpy())
                all_targets.append(lf_target.cpu().numpy())
        
        # Concatenate predictions and targets
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate additional metrics
        mae = np.mean(np.abs(all_preds - all_targets))
        mse = np.mean(np.square(all_preds - all_targets))
        
        # Calculate MAPE with safeguard against division by zero
        epsilon = 1e-10
        mape = np.mean(np.abs((all_targets - all_preds) / (np.abs(all_targets) + epsilon))) * 100
        
        return {
            "loss": np.mean(losses),
            "mae": mae,
            "mse": mse,
            "mape": mape
        }
    
    def predict(self, data_loader):
        """
        Generate predictions for the given data.
        Args:
            data_loader: DataLoader containing input data
        Returns:
            Dictionary containing predictions and actual targets
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Unpack batch data
                if len(batch) == 4:  # Includes client_id
                    hf_input, lf_input, lf_target, client_id = [
                        b.to(self.device) for b in batch
                    ]
                else:  # No client_id
                    hf_input, lf_input, lf_target = [
                        b.to(self.device) for b in batch
                    ]
                    client_id = None
                
                # Forward pass
                outputs = self.model(hf_input, lf_input, client_id)
                preds = outputs["lf_pred"]
                
                # Store predictions and targets
                all_preds.append(preds.cpu().numpy())
                all_targets.append(lf_target.cpu().numpy())
        
        # Concatenate all predictions and targets
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        return {
            "predictions": all_preds,
            "targets": all_targets
        }
