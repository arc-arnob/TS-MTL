import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class TSDiffGradBalTrainer:
    """Trainer for TSDiff model."""
    def __init__(self, model, diffusion_process, learning_rate=1e-3, device=None):
        self.model = model
        self.diffusion_process = diffusion_process
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.diffusion_process.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Initialize client weights for gradient balancing
        self.client_weights = {}
        self.grad_weight_strategy = 'cosine'
        self.grad_clip = 1.0
    
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
        for param_idx, _ in enumerate(self.model.parameters()):
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
    
    def train_step(self, batch):
        """Single training step with gradient balancing support."""
        if len(batch) == 2:
            x, client_id = batch
        else:
            x = batch[0]
            client_id = None
            
        x = x.to(self.device)
        batch_size = x.shape[0]
        
        # Check if we have multiple clients in this batch
        if client_id is not None and isinstance(client_id, torch.Tensor) and client_id.numel() > 1:
            unique_clients = torch.unique(client_id)
            
            if len(unique_clients) > 1:
                # Multiple clients - apply gradient balancing
                client_batches = {}
                
                for c in unique_clients:
                    idx = (client_id == c).nonzero(as_tuple=True)[0]
                    client_batches[c.item()] = x[idx]
                
                # Compute gradients for each client separately
                client_grads = {}
                client_losses = {}
                
                for cid, client_x in client_batches.items():
                    self.optimizer.zero_grad()
                    
                    # Sample noise and timesteps for this client
                    client_batch_size = client_x.shape[0]
                    noise = torch.randn_like(client_x)
                    timesteps = torch.randint(0, self.diffusion_process.num_timesteps, 
                                            (client_batch_size,), device=self.device)
                    
                    # Forward diffusion
                    x_noisy = self.diffusion_process.q_sample(client_x, timesteps, noise)
                    
                    # Predict noise
                    predicted_noise = self.model(x_noisy, timesteps)
                    
                    # Calculate loss
                    loss = self.criterion(predicted_noise, noise)
                    
                    # Backward pass to compute gradients
                    loss.backward()
                    
                    # Store gradients
                    model_params = list(self.model.parameters())
                    client_grads[cid] = [p.grad.clone() if p.grad is not None else None 
                                    for p in model_params]
                    client_losses[cid] = loss.item()
                
                # Apply gradient balancing
                weighted_grads = self.compute_weighted_gradients_capped(client_grads, client_losses)
                
                # Apply weighted gradients to model parameters
                self.optimizer.zero_grad()
                for param, grad in zip(self.model.parameters(), weighted_grads):
                    if grad is not None:
                        param.grad = grad
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Update parameters
                self.optimizer.step()
                
                # Calculate overall loss for reporting
                total_loss = sum(client_losses.values()) / len(client_losses)
                return {"loss": total_loss}
            
        # Single client or no client_id - standard training
        # Sample noise and timesteps
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, self.diffusion_process.num_timesteps, (batch_size,), device=self.device)
        
        # Forward diffusion
        x_noisy = self.diffusion_process.q_sample(x, timesteps, noise)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, timesteps)
        
        # Calculate loss
        loss = self.criterion(predicted_noise, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.optimizer.step()
        
        return {"loss": loss.item()}
    
    def fit(self, train_loader, val_loader=None, epochs=50, verbose=True):
        """Train the model."""
        history = {"train_loss": []}
        if val_loader:
            history["val_loss"] = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                step_result = self.train_step(batch)
                train_losses.append(step_result["loss"])
                
                if verbose and batch_idx % 100 == 0 and batch_idx > 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {np.mean(train_losses[-100:]):.4f}")
            
            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            if val_loader:
                val_loss = self.evaluate_loss(val_loader)
                history["val_loss"].append(val_loss)
            
            # Print epoch summary
            if verbose:
                elapsed_time = time.time() - start_time
                log_msg = f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Time: {elapsed_time:.2f}s"
                if val_loader:
                    log_msg += f", Val Loss: {val_loss:.4f}"
                print(log_msg)
        
        return history
    
    def evaluate_loss(self, data_loader):
        """Evaluate the model loss."""
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 2:
                    x, _ = batch
                else:
                    x = batch[0]
                
                x = x.to(self.device)
                batch_size = x.shape[0]
                
                noise = torch.randn_like(x)
                timesteps = torch.randint(0, self.diffusion_process.num_timesteps, (batch_size,), device=self.device)
                
                x_noisy = self.diffusion_process.q_sample(x, timesteps, noise)
                predicted_noise = self.model(x_noisy, timesteps)
                
                loss = self.criterion(predicted_noise, noise)
                losses.append(loss.item())
        
        return np.mean(losses)
    
    def predict_multihorizon(self, data_loader, forecast_horizon, guidance_scale=1.0, autoregressive=False):
        """Generate predictions for configurable forecast horizons."""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:  # Test mode: (multivariate_context, lf_target, client_id)
                    multivariate_context, lf_target, client_id = batch
                    multivariate_context = multivariate_context.to(self.device)
                    lf_target = lf_target.to(self.device)
                    
                    batch_size, context_len, total_features = multivariate_context.shape
                    lf_features = lf_target.shape[2]
                    lf_start_idx = total_features - lf_features
                    
                    if autoregressive:
                        # Autoregressive prediction: predict one step at a time
                        predictions = []
                        current_context = multivariate_context.clone()
                        
                        for step in range(forecast_horizon):
                            # Predict next single step
                            forecast_placeholder = torch.zeros(batch_size, 1, total_features, device=self.device)
                            full_sequence = torch.cat([current_context, forecast_placeholder], dim=1)
                            
                            # Add noise and denoise
                            noise = torch.randn_like(full_sequence) * 0.1
                            noisy_sequence = full_sequence + noise
                            timesteps = torch.zeros(batch_size, device=self.device, dtype=torch.long)
                            denoised = self.model(noisy_sequence, timesteps)
                            
                            # Extract the predicted next step
                            next_step_multivariate = denoised[:, -1:, :]  # Last timestep
                            next_step_lf = next_step_multivariate[:, :, lf_start_idx:]  # LF part only
                            predictions.append(next_step_lf)
                            
                            # Update context for next prediction (use predicted values)
                            # For simplicity, just extend context (could also use sliding window)
                            current_context = torch.cat([current_context[:, 1:, :], next_step_multivariate], dim=1)
                        
                        forecast_lf = torch.cat(predictions, dim=1)
                    else:
                        # Direct multi-step prediction (current approach)
                        forecast_placeholder = torch.zeros(batch_size, forecast_horizon, total_features, device=self.device)
                        full_sequence = torch.cat([multivariate_context, forecast_placeholder], dim=1)
                        
                        noise = torch.randn_like(full_sequence) * 0.1
                        noisy_sequence = full_sequence + noise
                        timesteps = torch.zeros(batch_size, device=self.device, dtype=torch.long)
                        denoised = self.model(noisy_sequence, timesteps)
                        
                        forecast_multivariate = denoised[:, context_len:, :]
                        forecast_lf = forecast_multivariate[:, :, lf_start_idx:]
                    
                    # Only take the forecast_horizon we want
                    forecast_lf = forecast_lf[:, :forecast_horizon, :]
                    target_trimmed = lf_target[:, :forecast_horizon, :]
                    
                    all_preds.append(forecast_lf.cpu().numpy())
                    all_targets.append(target_trimmed.cpu().numpy())
                else:
                    continue
        
        if not all_preds:
            return {"predictions": np.array([]), "targets": np.array([])}
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        return {"predictions": all_preds, "targets": all_targets}
    
    def evaluate_multihorizon(self, data_loader, forecast_horizon, autoregressive=False):
        """Evaluate model for specific forecast horizon."""
        results = self.predict_multihorizon(data_loader, forecast_horizon, autoregressive=autoregressive)
        
        if len(results["predictions"]) == 0:
            return {
                "loss": float('inf'),
                "mae": float('inf'),
                "mse": float('inf'),
                "mape": float('inf'),
                "forecast_horizon": forecast_horizon
            }
        
        preds = results["predictions"]
        targets = results["targets"]
        
        # Calculate metrics
        mae = np.mean(np.abs(preds - targets))
        mse = np.mean(np.square(preds - targets))
        
        # Calculate MAPE with safeguard
        epsilon = 1e-10
        mape = np.mean(np.abs((targets - preds) / (np.abs(targets) + epsilon))) * 100
        
        return {
            "loss": mse,
            "mae": mae,
            "mse": mse,
            "mape": mape,
            "forecast_horizon": forecast_horizon
        }
    
    def evaluate(self, data_loader, forecast_horizon=16):
        """Evaluate the model with configurable forecast horizon."""
        return self.evaluate_multihorizon(data_loader, forecast_horizon, autoregressive=False)