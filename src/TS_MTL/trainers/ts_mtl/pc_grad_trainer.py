import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

#####################################################
# PCGrad Trainer Implementation
#####################################################

class PCGradTrainer:
    """
    Trainer with PCGrad (Projecting Conflicting Gradients) approach.
    This mitigates negative interference between task gradients.
    """
    def __init__(self, model, learning_rate=0.001, device=None, grad_clip=1.0):
        self.model = model
        self.learning_rate = learning_rate
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.grad_clip = grad_clip
    
    def project_conflicting_gradients(self, grads_by_param):
        """
        Apply PCGrad - project each gradient onto the normal plane of other gradients
        when they conflict (have negative cosine similarity).
        
        Args:
            grads_by_param: Dictionary mapping parameter indices to lists of client gradients
            
        Returns:
            Dictionary of modified gradients with reduced interference
        """
        result = {}
        
        # Process each parameter separately to avoid dimension mismatch
        for param_idx, client_grads in grads_by_param.items():
            # Skip if we only have one client's gradient for this parameter
            if len(client_grads) <= 1:
                if client_grads:  # If there's exactly one gradient
                    result[param_idx] = client_grads[0]
                continue
                
            # Clone gradients to avoid modifying the originals
            projected_grads = [g.clone() for g in client_grads]
            
            # Get flattened versions for similarity calculation
            flat_grads = [g.reshape(-1) for g in client_grads]
            
            # Apply projections between all pairs of gradients
            for i in range(len(projected_grads)):
                for j in range(len(client_grads)):
                    if i == j:
                        continue
                    
                    # Get the flattened gradients
                    g_i_flat = flat_grads[i]
                    g_j_flat = flat_grads[j]
                    
                    # Skip if either gradient is too small
                    g_i_norm = torch.norm(g_i_flat)
                    g_j_norm = torch.norm(g_j_flat)
                    
                    if g_i_norm < 1e-8 or g_j_norm < 1e-8:
                        continue
                    
                    # Calculate cosine similarity manually
                    dot_product = torch.dot(g_i_flat, g_j_flat)
                    cos_sim = dot_product / (g_i_norm * g_j_norm)
                    
                    # Project if gradients conflict (negative cosine similarity)
                    if cos_sim < 0:
                        # Calculate the projection of g_i onto g_j
                        projection = (dot_product / (g_j_norm * g_j_norm)) * g_j_flat
                        
                        # Reshape back to original shape and subtract from g_i
                        projection_reshaped = projection.reshape(projected_grads[i].shape)
                        projected_grads[i] = projected_grads[i] - projection_reshaped
            
            # Average the projected gradients for this parameter
            result[param_idx] = sum(projected_grads) / len(projected_grads)
        
        return result
    
    def train_step(self, batch):
        """Perform a single training step with PCGrad."""
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
        
        # Sort batch samples by client_id
        if client_id is not None and client_id.numel() > 1:
            client_batches = {}
            unique_clients = torch.unique(client_id)
            
            for c in unique_clients:
                idx = (client_id == c).nonzero(as_tuple=True)[0]
                client_batches[c.item()] = (
                    hf_input[idx], 
                    lf_input[idx], 
                    lf_target[idx], 
                    client_id[idx]
                )
            
            # Compute gradients for each client separately
            client_grads = {}
            client_losses = {}
            
            for cid, (c_hf, c_lf, c_target, c_id) in client_batches.items():
                self.optimizer.zero_grad()
                
                # Forward pass for this client
                outputs = self.model(c_hf, c_lf, c_id)
                loss = self.criterion(outputs["lf_pred"], c_target)
                
                # Backward pass to compute gradients
                loss.backward()
                
                # Store gradients for shared parameters
                shared_params = self.model.get_shared_parameters()
                client_grads[cid] = [p.grad.clone() if p.grad is not None else None 
                                   for p in shared_params]
                client_losses[cid] = loss.item()
            
            # Apply PCGrad to the shared parameters
            self.optimizer.zero_grad()
            
            # Reorganize gradients by parameter instead of by client
            # to avoid dimension mismatch issues
            grads_by_param = {}
            shared_params = self.model.get_shared_parameters()
            
            for param_idx in range(len(shared_params)):
                grads_by_param[param_idx] = []
                for cid in client_batches:
                    if client_grads[cid][param_idx] is not None:
                        grads_by_param[param_idx].append(client_grads[cid][param_idx])
            
            # Apply PCGrad to modify gradients, processing each parameter separately
            modified_grads = self.project_conflicting_gradients(grads_by_param)
            
            # Apply the modified gradients to the shared parameters
            for param_idx, grad in modified_grads.items():
                shared_params[param_idx].grad = grad
            
            # Update client-specific parameters individually
            for cid in client_batches:
                # Forward pass for this client again
                c_hf, c_lf, c_target, c_id = client_batches[cid]
                outputs = self.model(c_hf, c_lf, c_id)
                loss = self.criterion(outputs["lf_pred"], c_target)
                
                # Backward pass but retain graph to avoid zeroing shared parameters' grads
                loss.backward(retain_graph=True)
                
                # Update only client-specific parameters
                for param in self.model.get_client_parameters(cid):
                    if param.grad is not None:
                        param.grad.data.clamp_(-self.grad_clip, self.grad_clip)
                
                # Now we need a targeted optimizer step for just these parameters
                with torch.no_grad():
                    for param in self.model.get_client_parameters(cid):
                        if param.grad is not None:
                            param.data.add_(param.grad, alpha=-self.learning_rate)
            
            # Now take the shared step
            self.optimizer.step()
            
            # Calculate the overall loss for reporting
            total_loss = sum(client_losses.values()) / len(client_losses)
            return {"loss": total_loss}
            
        else:
            # Simple case - single client or no client_id
            self.optimizer.zero_grad()
            outputs = self.model(hf_input, lf_input, client_id)
            loss = self.criterion(outputs["lf_pred"], lf_target)
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            return {"loss": loss.item()}
    
    def fit(self, train_loader, val_loader=None, epochs=50, early_stopping_patience=10, verbose=True):
        """
        Train the model using PCGrad.
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
        client_metrics = defaultdict(lambda: {"loss": [], "preds": [], "targets": []})
        
        with torch.no_grad():
            for batch in data_loader:
                # Unpack batch data
                if len(batch) == 4:  # Includes client_id
                    hf_input, lf_input, lf_target, client_id = [
                        b.to(self.device) for b in batch
                    ]
                    
                    # Group by client for per-client metrics
                    if client_id.numel() > 1:
                        unique_clients = torch.unique(client_id)
                        for c in unique_clients:
                            idx = (client_id == c).nonzero(as_tuple=True)[0]
                            c_hf = hf_input[idx]
                            c_lf = lf_input[idx]
                            c_target = lf_target[idx]
                            c_id = client_id[idx]
                            
                            outputs = self.model(c_hf, c_lf, c_id)
                            preds = outputs["lf_pred"]
                            loss = self.criterion(preds, c_target)
                            
                            # Store client-specific metrics
                            cid = c.item()
                            client_metrics[cid]["loss"].append(loss.item())
                            client_metrics[cid]["preds"].append(preds.cpu().numpy())
                            client_metrics[cid]["targets"].append(c_target.cpu().numpy())
                    else:
                        outputs = self.model(hf_input, lf_input, client_id)
                        preds = outputs["lf_pred"]
                        loss = self.criterion(preds, lf_target)
                        
                        # Store metrics
                        cid = client_id.item()
                        client_metrics[cid]["loss"].append(loss.item())
                        client_metrics[cid]["preds"].append(preds.cpu().numpy())
                        client_metrics[cid]["targets"].append(lf_target.cpu().numpy())
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
        
        # Process client-specific metrics
        client_results = {}
        for cid, metrics in client_metrics.items():
            preds = np.concatenate(metrics["preds"], axis=0)
            targets = np.concatenate(metrics["targets"], axis=0)
            
            # Calculate additional metrics
            mae = np.mean(np.abs(preds - targets))
            mse = np.mean(np.square(preds - targets))
            
            # Calculate MAPE with safeguard against division by zero
            epsilon = 1e-10
            mape = np.mean(np.abs((targets - preds) / (np.abs(targets) + epsilon))) * 100
            
            client_results[cid] = {
                "loss": np.mean(metrics["loss"]),
                "mae": mae,
                "mse": mse,
                "mape": mape
            }
        
        # If there were no client IDs, use the aggregated metrics
        if not client_results:
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
                "mape": mape,
                "client_results": {}
            }
        
        # Calculate overall metrics from client results
        overall_metrics = {
            "loss": np.mean([r["loss"] for r in client_results.values()]),
            "mae": np.mean([r["mae"] for r in client_results.values()]),
            "mse": np.mean([r["mse"] for r in client_results.values()]),
            "mape": np.mean([r["mape"] for r in client_results.values()]),
            "client_results": client_results
        }
        
        return overall_metrics
    
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