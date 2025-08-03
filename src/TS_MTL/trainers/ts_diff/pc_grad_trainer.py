import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class TSDiffPCGradTrainer:
    """Trainer for TSDiff model."""
    def __init__(self, model, diffusion_process, learning_rate=1e-3, device=None):
        self.model = model
        self.diffusion_process = diffusion_process
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.diffusion_process.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def project_conflicting_gradients(self, grads_by_param):
        """
        Apply PCGrad - project each gradient onto the normal plane of other gradients
        when they conflict (have negative cosine similarity).
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
        """Single training step with PCGrad support."""
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
                # Multiple clients - apply PCGrad
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
                
                # Apply PCGrad
                self.optimizer.zero_grad()
                
                # Reorganize gradients by parameter
                grads_by_param = {}
                model_params = list(self.model.parameters())
                
                for param_idx in range(len(model_params)):
                    grads_by_param[param_idx] = []
                    for cid in client_batches:
                        if client_grads[cid][param_idx] is not None:
                            grads_by_param[param_idx].append(client_grads[cid][param_idx])
                
                # Apply PCGrad to modify gradients
                modified_grads = self.project_conflicting_gradients(grads_by_param)
                
                # Apply the modified gradients to model parameters
                for param_idx, grad in modified_grads.items():
                    if param_idx < len(model_params):
                        model_params[param_idx].grad = grad
                
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
