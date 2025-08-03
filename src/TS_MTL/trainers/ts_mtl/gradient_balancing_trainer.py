import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

#####################################################
# Gradient Balancing Trainer
#####################################################

class GradientBalancingTrainer:
    """
    Trainer with gradient balancing to handle conflicting gradients
    between different client tasks.
    """
    def __init__(self, model, learning_rate=0.001, device=None, 
                 grad_weight_strategy='cosine', grad_clip=1.0):
        self.model = model
        self.learning_rate = learning_rate
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.grad_clip = grad_clip
        self.grad_weight_strategy = grad_weight_strategy
        
        # Initialize client weights for gradient balancing
        if hasattr(model, 'client_ids') and model.client_ids:
            self.client_weights = {client_id: 1.0 for client_id in model.client_ids}
        else:
            self.client_weights = {}
    
    def compute_grad_cosine_similarity(self, grads1, grads2):
        """Compute cosine similarity between two gradient vectors"""
        # Flatten gradients
        vec1, vec2 = [], []
        for g1, g2 in zip(grads1, grads2):
            if g1 is not None and g2 is not None:
                vec1.append(g1.view(-1))
                vec2.append(g2.view(-1))
        
        if not vec1 or not vec2:
            return 0.0
            
        vec1 = torch.cat(vec1)
        vec2 = torch.cat(vec2)
        
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
        return cos_sim

    def compute_weighted_gradients(self, client_grads, client_losses):
        """Compute weighted gradients based on similarity and losses"""
        num_clients = len(client_grads)
        if num_clients <= 1: # If only one client is found.
            if client_grads:
                only_client_id = next(iter(client_grads.keys()))
                return client_grads[only_client_id]
            return None
            
        # Initialize weights based on losses (higher loss = higher weight)
        loss_weights = {cid: loss for cid, loss in client_losses.items()}
        total_loss = sum(loss_weights.values())
        if total_loss > 0:
            loss_weights = {cid: loss / total_loss for cid, loss in loss_weights.items()}
        
        # Compute similarities between client gradients
        similarities = {}
        client_ids = list(client_grads.keys())
        
        for i, cid1 in enumerate(client_ids):
            similarities[cid1] = {}
            for cid2 in client_ids:
                if cid1 == cid2:
                    similarities[cid1][cid2] = 1.0
                elif cid2 in similarities and cid1 in similarities[cid2]:
                    similarities[cid1][cid2] = similarities[cid2][cid1]
                else:
                    sim = self.compute_grad_cosine_similarity(client_grads[cid1], client_grads[cid2])
                    similarities[cid1][cid2] = sim
        
        # Adjust weights based on conflicts (negative similarities)
        adjusted_weights = self.client_weights.copy()
        
        if self.grad_weight_strategy == 'cosine':
            # Compute weight adjustments based on cosine similarity
            for cid1 in client_ids:
                for cid2 in client_ids:
                    if cid1 != cid2:
                        sim = similarities[cid1][cid2]
                        # Reduce weight for conflicting gradients
                        if sim < 0:
                            conflict_strength = abs(sim)
                            adjustment = 1.0 - (conflict_strength * 0.5)  # Reduce up to 50% for complete conflict
                            adjusted_weights[cid1] *= adjustment
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        normalized_weights = {cid: w / total_weight for cid, w in adjusted_weights.items()}
        
        # Apply weights to gradients
        weighted_grads = []
        for param_idx, _ in enumerate(self.model.get_shared_parameters()):
            weighted_grad = None
            for cid, grads in client_grads.items():
                if grads[param_idx] is not None:
                    if weighted_grad is None:
                        weighted_grad = grads[param_idx] * normalized_weights[cid]
                    else:
                        weighted_grad += grads[param_idx] * normalized_weights[cid]
            weighted_grads.append(weighted_grad)
        
        # Update client weights for next iteration
        self.client_weights = adjusted_weights
        
        return weighted_grads
    
    def compute_weighted_gradients_capped(self, client_grads, client_losses):
        """Compute weighted gradients based on similarity and losses"""
        num_clients = len(client_grads)
        if num_clients <= 1:  # If only one client is found
            if client_grads:
                only_client_id = next(iter(client_grads.keys()))
                return client_grads[only_client_id]
            return None
        
        client_ids = list(client_grads.keys())
        
        # Ensure all clients have weights
        for cid in client_ids:
            if cid not in self.client_weights:
                # print(f"Adding missing client {cid} to weights dictionary")
                self.client_weights[cid] = 1.0
        
        # Reset weights if any are too small
        min_weight = min(self.client_weights.values()) if self.client_weights else 0
        if min_weight < 1e-3:  # If any weight gets too small, reset all weights
            print("Resetting client weights due to small values")
            for cid in self.client_weights:
                self.client_weights[cid] = 1.0
        
        # Compute loss-based weights
        loss_weights = {cid: max(loss, 0.1) for cid, loss in client_losses.items()}
        total_loss = sum(loss_weights.values())
        if total_loss > 0:
            loss_weights = {cid: loss / total_loss for cid, loss in loss_weights.items()}
        
        # Compute similarities between client gradients
        similarities = {}
        
        for i, cid1 in enumerate(client_ids):
            similarities[cid1] = {}
            for cid2 in client_ids:
                if cid1 == cid2:
                    similarities[cid1][cid2] = 1.0
                elif cid2 in similarities and cid1 in similarities[cid2]:
                    similarities[cid1][cid2] = similarities[cid2][cid1]
                else:
                    sim = self.compute_grad_cosine_similarity(client_grads[cid1], client_grads[cid2])
                    similarities[cid1][cid2] = sim
        
        # Calculate adjustment factors
        adjustment_factors = {}
        if self.grad_weight_strategy == 'cosine':
            for cid1 in client_ids:
                conflict_adjustment = 1.0
                conflict_count = 0
                
                for cid2 in client_ids:
                    if cid1 != cid2:
                        sim = similarities[cid1][cid2]
                        if sim < 0:
                            conflict_strength = abs(sim)
                            adjustment = max(0.7, 1.0 - (conflict_strength * 0.3))
                            conflict_adjustment *= adjustment
                            conflict_count += 1
                
                if conflict_count > 0:
                    conflict_adjustment = max(0.5, conflict_adjustment)
                
                adjustment_factors[cid1] = conflict_adjustment
        else:
            # Default: equal weighting
            for cid in client_ids:
                adjustment_factors[cid] = 1.0
        
        # Combine base weights with adjustment factors
        adjusted_weights = {}
        for cid in client_ids:
            current_weight = self.client_weights[cid]
            loss_weight = loss_weights.get(cid, 1.0/num_clients)  # Default if not found
            adjustment = adjustment_factors.get(cid, 1.0)  # Default if not found
            
            new_weight = 0.9 * current_weight + 0.1 * (loss_weight * adjustment)
            adjusted_weights[cid] = new_weight
        
        # Ensure minimum weight
        min_weight = 0.2 / num_clients
        for cid in adjusted_weights:
            adjusted_weights[cid] = max(adjusted_weights[cid], min_weight)
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        normalized_weights = {cid: w / total_weight for cid, w in adjusted_weights.items()}
        
        # Apply weights to gradients
        weighted_grads = []
        for param_idx, _ in enumerate(self.model.get_shared_parameters()):
            weighted_grad = None
            for cid, grads in client_grads.items():
                if grads[param_idx] is not None:
                    if weighted_grad is None:
                        weighted_grad = grads[param_idx] * normalized_weights[cid]
                    else:
                        weighted_grad += grads[param_idx] * normalized_weights[cid]
            weighted_grads.append(weighted_grad)
        
        # Update client weights for next iteration
        self.client_weights = adjusted_weights
        # print(self.client_weights)
        return weighted_grads
    
    def train_step(self, batch):
        """Perform a single training step with gradient balancing."""
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
                
                # Save gradients
                client_grads[cid] = [p.grad.clone() if p.grad is not None else None 
                                     for p in self.model.get_shared_parameters()]
                client_losses[cid] = loss.item()
            
            # Compute weighted gradients
            # print("ERROR!!!!: ", len(client_grads))
            
            weighted_grads = self.compute_weighted_gradients_capped(client_grads, client_losses)
            
            # Apply weighted gradients to shared parameters
            self.optimizer.zero_grad()
            for param, grad in zip(self.model.get_shared_parameters(), weighted_grads):
                if grad is not None:
                    param.grad = grad
            
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            return {"loss": loss.item()}
    
    def fit(self, train_loader, val_loader=None, epochs=50, early_stopping_patience=10, verbose=True):
        """
        Train the model with gradient balancing.
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
            "val_loss": [] if val_loader else None,
            "client_weights": []
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
            
            # Record client weights for this epoch
            history["client_weights"].append(self.client_weights.copy())
            
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
                
                # Print client weights
                weights_str = ", ".join([f"Client {cid}: {w:.2f}" for cid, w in sorted(self.client_weights.items())])
                log_msg += f", Weights: [{weights_str}]"
                
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