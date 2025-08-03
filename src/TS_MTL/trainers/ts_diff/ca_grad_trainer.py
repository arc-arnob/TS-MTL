import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class TSDiffCAGradTrainer:
    """Trainer for TSDiff model with CAGrad gradient handling."""
    def __init__(self, model, diffusion_process, learning_rate=1e-3, device=None, cagrad_c=0.4):
        self.model = model
        self.diffusion_process = diffusion_process
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cagrad_c = cagrad_c
        
        self.model.to(self.device)
        self.diffusion_process.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def cagrad(self, grad_vec, num_tasks):
        """
        CAGrad algorithm implementation
        Args:
            grad_vec: [num_tasks, dim] tensor of task gradients
            num_tasks: number of tasks
        Returns:
            Conflict-averse gradient
        """
        grads = grad_vec
        
        # Compute Gram matrix
        GG = grads.mm(grads.t()).cpu()
        scale = (torch.diag(GG) + 1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)
        
        # Initialize weights
        w = torch.zeros(num_tasks, 1, requires_grad=True)
        if num_tasks == 50:
            w_opt = torch.optim.SGD([w], lr=50, momentum=0.5)
        else:
            w_opt = torch.optim.SGD([w], lr=25, momentum=0.5)
        
        c = (gg + 1e-4).sqrt() * self.cagrad_c
        
        w_best = None
        obj_best = np.inf
        
        # Optimize weights
        for i in range(21):
            w_opt.zero_grad()
            ww = torch.softmax(w, 0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < 20:
                obj.backward()
                w_opt.step()
        
        ww = torch.softmax(w_best, 0)
        gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
        
        lmbda = c.view(-1) / (gw_norm + 1e-4)
        g = ((1/num_tasks + ww * lmbda).view(-1, 1).to(grads.device) * grads).sum(0) / (1 + self.cagrad_c**2)
        
        return g
    
    def train_step(self, batch):
        """Single training step with CAGrad support."""
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
                # Multiple clients - apply CAGrad
                client_batches = {}
                
                for c in unique_clients:
                    idx = (client_id == c).nonzero(as_tuple=True)[0]
                    client_batches[c.item()] = x[idx]
                
                # Compute gradients for each client separately
                task_grads = []
                task_losses = []
                
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
                    
                    # Convert gradients to vector
                    grad_vec = torch.cat([
                        p.grad.view(-1) for p in self.model.parameters() 
                        if p.grad is not None
                    ])
                    task_grads.append(grad_vec)
                    task_losses.append(loss.item())
                
                # Stack gradients
                grad_matrix = torch.stack(task_grads, dim=0)  # [num_tasks, dim]
                
                # Apply CAGrad
                cagrad_gradient = self.cagrad(grad_matrix, len(task_grads))
                
                # Apply the CAGrad gradient to model parameters
                self.optimizer.zero_grad()
                idx = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        param_size = param.numel()
                        param.grad = cagrad_gradient[idx:idx+param_size].view_as(param)
                        idx += param_size
                
                # Update parameters
                self.optimizer.step()
                
                # Calculate overall loss for reporting
                total_loss = sum(task_losses) / len(task_losses)
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