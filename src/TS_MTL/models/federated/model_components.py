"""
Model Components for Federated Learning

This module contains the neural network components used in federated learning systems.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np

# Federated Model Components

class LSTMEncoder(nn.Module):
    """LSTM-based encoder for high-frequency data."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the encoder.
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            output: LSTM output tensor of shape (batch_size, seq_len, hidden_dim)
            (h_n, c_n): Final hidden and cell states
        """
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)


class TemporalAttention(nn.Module):
    """Temporal attention mechanism."""
    
    def __init__(self, query_dim: int, key_dim: int):
        super().__init__()
        self.scale = 1.0 / (key_dim ** 0.5)
        self.query_proj = nn.Linear(query_dim, key_dim)
        self.key_proj = nn.Linear(key_dim, key_dim)
        self.value_proj = nn.Linear(key_dim, key_dim)
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Calculate attention-weighted context.
        Args:
            query: Query tensor (decoder state)
            keys: Key tensors (encoder outputs)
        Returns:
            context: Context vector
        """
        # Project query and keys
        query = self.query_proj(query)  # (batch_size, 1, key_dim)
        keys = self.key_proj(keys)      # (batch_size, seq_len, key_dim)
        values = self.value_proj(keys)  # (batch_size, seq_len, key_dim)
        
        # Calculate attention scores
        scores = torch.bmm(query, keys.transpose(1, 2)) * self.scale  # (batch_size, 1, seq_len)
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=2)  # (batch_size, 1, seq_len)
        
        # Calculate context vector
        context = torch.bmm(attn_weights, values)  # (batch_size, 1, key_dim)
        
        return context

class LSTMDecoder(nn.Module):
    """LSTM-based decoder for low-frequency data."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attn = TemporalAttention(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, encoder_outputs: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass through the decoder.
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            encoder_outputs: Outputs from the encoder
            hidden: Initial hidden state (optional)
        Returns:
            output: Predicted values of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize output tensor to store results
        outputs = torch.zeros(batch_size, seq_len, self.fc.out_features).to(x.device)
        
        # Initial hidden state (if not provided)
        h_t = hidden
        
        # Process each time step
        for t in range(seq_len):
            # Get input for this time step
            x_t = x[:, t:t+1, :]  # (batch_size, 1, input_dim)
            
            # Run LSTM cell
            if h_t is None:
                lstm_out, h_t = self.lstm(x_t)
            else:
                lstm_out, h_t = self.lstm(x_t, h_t)
            
            # Apply attention
            context = self.attn(lstm_out, encoder_outputs)
            
            # Combine context with LSTM output
            combined = lstm_out + context
            
            # Generate output for this time step
            out_t = self.fc(combined.squeeze(1))  # (batch_size, output_dim)
            
            # Store in the right position
            outputs[:, t, :] = out_t
        
        return outputs

class TemporalAttention(nn.Module):
    """Temporal attention mechanism."""
    def __init__(self, query_dim, key_dim):
        super().__init__()
        self.scale = 1.0 / (key_dim ** 0.5)
        self.query_proj = nn.Linear(query_dim, key_dim)
        self.key_proj = nn.Linear(key_dim, key_dim)
        self.value_proj = nn.Linear(key_dim, key_dim)
        
    def forward(self, query, keys):
        """
        Calculate attention-weighted context.
        Args:
            query: Query tensor (decoder state)
            keys: Key tensors (encoder outputs)
        Returns:
            context: Context vector
        """
        # Project query and keys
        query = self.query_proj(query)  # (batch_size, 1, key_dim)
        keys = self.key_proj(keys)      # (batch_size, seq_len, key_dim)
        values = self.value_proj(keys)  # (batch_size, seq_len, key_dim)
        
        # Calculate attention scores
        scores = torch.bmm(query, keys.transpose(1, 2)) * self.scale  # (batch_size, 1, seq_len)
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=2)  # (batch_size, 1, seq_len)
        
        # Calculate context vector
        context = torch.bmm(attn_weights, values)  # (batch_size, 1, key_dim)
        
        return context

class BahdanauTemporalAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism for LSTM encoder–decoder.
    
    Given a decoder hidden state (query) and a sequence of encoder hidden 
    states (keys), this module computes alignment energies, converts them 
    into a softmax weight distribution, and then produces a weighted sum 
    of the encoder outputs as the context vector.
    
    Args:
        hidden_dim (int): Dimensionality of the decoder LSTM hidden state and encoder output.
        attention_dim (int): Dimensionality of the intermediate attention representation.
                               If not specified, you can set it to a value such as hidden_dim // 2.
    """
    def __init__(self, hidden_dim, attention_dim=None):
        super().__init__()
        if attention_dim is None:
            attention_dim = hidden_dim // 2
        
        # Linear transformations for decoder state (query) and encoder output (key)
        self.W_query = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.W_key = nn.Linear(hidden_dim, attention_dim, bias=False)
        # This layer projects the combined representation to a scalar energy
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, query, keys):
        """
        Args:
            query (Tensor): Decoder hidden state at time t-1 with shape (batch_size, hidden_dim)
                            or (batch_size, 1, hidden_dim).
            keys (Tensor): Encoder hidden states with shape (batch_size, seq_len, hidden_dim).

        Returns:
            context (Tensor): The context vector computed as the weighted sum of keys,
                              with shape (batch_size, 1, hidden_dim).
            attn_weights (Tensor): The attention weights, shape (batch_size, seq_len).
        """
        # Ensure query has shape (batch_size, 1, hidden_dim)
        if query.dim() == 2:
            query = query.unsqueeze(1)
        
        # Transform query and keys into an "attention space"
        query_transformed = self.W_query(query)       # (B, 1, attention_dim)
        keys_transformed = self.W_key(keys)             # (B, T, attention_dim)
        
        # Compute energy scores for each encoder output:
        # Broadcasting the query_transformed over the time dimension.
        energy = torch.tanh(query_transformed + keys_transformed)  # (B, T, attention_dim)
        energy = self.v(energy).squeeze(-1)                        # (B, T)
        
        # Convert energy scores to attention weights
        attn_weights = torch.softmax(energy, dim=1)                # (B, T)
        
        # Compute context vector as the weighted sum of the encoder outputs (keys)
        context = torch.bmm(attn_weights.unsqueeze(1), keys)       # (B, 1, hidden_dim)
        
        return context #, attn_weights

class LSTMDecoderWithBahdanauAttention(nn.Module):
    """LSTM-based decoder for low-frequency data with Bahdanau-style attention on HF encoder outputs."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0, attention_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn = BahdanauTemporalAttention(hidden_dim, attention_dim=attention_dim or hidden_dim // 2)
        
        # Decoder input will now be: [LF_t || context], so increase input dim
        self.lstm = nn.LSTM(
            input_size=input_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, encoder_outputs, hidden=None):
        """
        Args:
            x: LF input tensor (B, T, input_dim)
            encoder_outputs: HF encoder output (B, T_enc, hidden_dim)
            hidden: (h_n, c_n), optional
        Returns:
            outputs: Predicted values (B, T, output_dim)
        """
        batch_size, seq_len, _ = x.size()
        outputs = torch.zeros(batch_size, seq_len, self.fc.out_features).to(x.device)

        h_t = hidden

        for t in range(seq_len):
            # LF input at time t
            x_t = x[:, t:t+1, :]  # (B, 1, input_dim)

            # Get context vector using attention from encoder outputs
            if h_t is None:
                query = torch.zeros(batch_size, self.hidden_dim).to(x.device)  # start with 0 query
            else:
                query = h_t[0][-1]  # use top-layer decoder hidden state as query (B, hidden_dim)

            context = self.attn(query, encoder_outputs)  # (B, 1, hidden_dim)

            # Concatenate LF input and context # Fix with Fusion Layer
            decoder_input = torch.cat([x_t, context], dim=2)  # (B, 1, input_dim + hidden_dim)

            # LSTM forward
            output, h_t = self.lstm(decoder_input, h_t)  # output: (B, 1, hidden_dim)

            # Final projection
            out_t = self.fc(output.squeeze(1))  # (B, output_dim)
            outputs[:, t, :] = out_t

        return outputs

class PrivateEncoder(nn.Module):
    """Privacy-preserving wrapper for LSTM encoder that adds DP noise to outputs."""
    def __init__(self, original_encoder, noise_scale=0.05, clip_bound=1.0):
        super().__init__()
        self.encoder = original_encoder
        self.noise_scale = noise_scale
        self.clip_bound = clip_bound
        
    def forward(self, x):
        # Original encoding
        encoder_outputs, encoder_hidden = self.encoder(x)
        
        # Apply element-wise clipping to bound sensitivity
        encoder_outputs_norm = torch.norm(encoder_outputs, dim=2, keepdim=True)
        scaling_factor = torch.min(
            torch.ones_like(encoder_outputs_norm),
            self.clip_bound / (encoder_outputs_norm + 1e-10)
        )
        encoder_outputs_clipped = encoder_outputs * scaling_factor
        
        # Add calibrated noise for differential privacy
        if self.training and self.noise_scale > 0:
            batch_size = x.size(0)
            noise = torch.randn_like(encoder_outputs_clipped) * self.noise_scale * self.clip_bound / np.sqrt(batch_size)
            private_outputs = encoder_outputs_clipped + noise
        else:
            # During evaluation, we can optionally skip adding noise
            private_outputs = encoder_outputs_clipped
        
        return encoder_outputs, encoder_hidden # private_outputs

class DPOptimizer:
    """Wrapper for any optimizer to add differential privacy to gradients."""
    def __init__(self, optimizer, noise_scale=0.1, max_grad_norm=1.0):
        self.optimizer = optimizer
        self.noise_scale = noise_scale
        self.max_grad_norm = max_grad_norm
        
    def zero_grad(self):
        """Clear gradients."""
        self.optimizer.zero_grad()
        
    def step(self):
        """Apply DP-SGD and then optimizer step."""
        # Get all parameter groups
        param_groups = self.optimizer.param_groups
        
        for group in param_groups:
            # Clip gradients for each parameter group
            torch.nn.utils.clip_grad_norm_(group['params'], self.max_grad_norm)
            
            # Add noise to gradients
            for p in group['params']:
                if p.requires_grad and p.grad is not None:
                    noise = torch.randn_like(p.grad) * self.noise_scale * self.max_grad_norm
                    p.grad += noise
        
        # Apply optimizer step
        self.optimizer.step()
        
    def state_dict(self):
        """Get state dict from optimizer."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict to optimizer."""
        self.optimizer.load_state_dict(state_dict)

class SecureAggregation:
    """Helper class for secure aggregation in federated learning."""
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def generate_masking_vectors(self, model_shape_dict, num_clients):
        """
        Generate random masking vectors for each client.
        The sum of all masks should be zero.
        
        Args:
            model_shape_dict: Dictionary mapping parameter names to their shapes
            num_clients: Number of clients participating in aggregation
            
        Returns:
            List of dictionaries containing masks for each client
        """
        masks = [{} for _ in range(num_clients)]
        
        # For each parameter in the model
        for param_name, param_shape in model_shape_dict.items():
            # For all but the last client, generate random masks
            for i in range(num_clients - 1):
                masks[i][param_name] = torch.randn(param_shape).to(self.device)
            
            # For the last client, make sure the sum of masks is zero
            masks[num_clients - 1][param_name] = -sum(mask[param_name] for mask in masks[:-1])
            
        return masks
    
    def apply_mask(self, model_update, mask):
        """
        Apply the masking vector to the model update.
        
        Args:
            model_update: Dictionary containing model updates
            mask: Dictionary containing masks
            
        Returns:
            Dictionary containing masked updates
        """
        masked_update = {}
        for param_name, param_update in model_update.items():
            masked_update[param_name] = param_update + mask[param_name]
        return masked_update
    
    def compute_model_shape_dict(self, model):
        """
        Compute the shape dictionary of a model's parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary mapping parameter names to their shapes
        """
        return {name: param.shape for name, param in model.state_dict().items()}



# FedProx Stuff
class ProximalTerm:
    """Helper class to calculate FedProx proximal term penalty."""
    def __init__(self, mu=0.01):
        """
        Initialize the ProximalTerm.
        Args:
            mu: Proximal term coefficient (regularization strength)
        """
        self.mu = mu
        self.global_params = None
        
    def set_global_params(self, model_state_dict):
        """Set the global model parameters for proximal term calculation."""
        self.global_params = model_state_dict
        
    def compute_proximal_term(self, model):
        """
        Compute the proximal term penalty between current model and global model.
        Returns the proximal term loss to be added to the client's objective.
        """
        if self.global_params is None:
            return 0.0
            
        proximal_term = 0.0
        for name, param in model.named_parameters():
            if name in self.global_params:
                proximal_term += torch.sum((param - self.global_params[name].detach()) ** 2)
                
        return 0.5 * self.mu * proximal_term
class DPOptimizerWithProximal:
    """Wrapper for any optimizer to add differential privacy to gradients with FedProx support."""
    def __init__(self, optimizer, proximal_calculator=None, noise_scale=0.1, max_grad_norm=1.0):
        self.optimizer = optimizer
        self.noise_scale = noise_scale
        self.max_grad_norm = max_grad_norm
        self.proximal_calculator = proximal_calculator
        
    def zero_grad(self):
        """Clear gradients."""
        self.optimizer.zero_grad()
        
    def step(self):
        """Apply DP-SGD and then optimizer step."""
        # Get all parameter groups
        param_groups = self.optimizer.param_groups
        
        for group in param_groups:
            # Clip gradients for each parameter group
            torch.nn.utils.clip_grad_norm_(group['params'], self.max_grad_norm)
            
            # Add noise to gradients
            for p in group['params']:
                if p.requires_grad and p.grad is not None:
                    noise = torch.randn_like(p.grad) * self.noise_scale * self.max_grad_norm
                    p.grad += noise
        
        # Apply optimizer step
        self.optimizer.step()
        
    def state_dict(self):
        """Get state dict from optimizer."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict to optimizer."""
        self.optimizer.load_state_dict(state_dict)


# SCAFFOLD STUFF
class ScaffoldControlVariate:
    """Helper class to manage SCAFFOLD control variates."""
    def __init__(self, model_structure):
        """
        Initialize control variates with same structure as model parameters.
        Args:
            model_structure: Dictionary mapping parameter names to their shapes
        """
        self.control_variates = {}
        for name, shape in model_structure.items():
            self.control_variates[name] = torch.zeros(shape)
    
    def get_dict(self):
        """Get control variates as dictionary."""
        return {name: tensor.clone() for name, tensor in self.control_variates.items()}
    
    def update_from_dict(self, new_variates):
        """Update control variates from dictionary."""
        for name in self.control_variates:
            if name in new_variates:
                self.control_variates[name] = new_variates[name].clone().detach()
    
    def to(self, device):
        """Move control variates to device."""
        for name in self.control_variates:
            self.control_variates[name] = self.control_variates[name].to(device)
        return self
    
    def zero_(self):
        """Zero out all control variates."""
        for name in self.control_variates:
            self.control_variates[name].zero_()


class DPOptimizerWithScaffold:
    """Wrapper for any optimizer to add differential privacy and SCAFFOLD control variates."""
    def __init__(self, optimizer, model, noise_scale=0.1, max_grad_norm=1.0):
        self.optimizer = optimizer  # Fixed: removed comma
        self.model = model  
        self.noise_scale = noise_scale
        self.max_grad_norm = max_grad_norm
        
    def zero_grad(self):
        """Clear gradients."""
        self.optimizer.zero_grad()
        
    def step(self, server_control_variate=None, client_control_variate=None):
        """
        Apply SCAFFOLD correction, DP-SGD and then optimizer step.
        Args:
            server_control_variate: Server control variate dict
            client_control_variate: Client control variate dict
        """
        # Apply SCAFFOLD control variate correction
        if server_control_variate is not None and client_control_variate is not None:
            # Create parameter name to parameter mapping
            param_dict = dict(self.model.named_parameters())
            
            for name, param in param_dict.items():
                if param.requires_grad and param.grad is not None:
                    # Apply SCAFFOLD correction: grad = grad - c_i + c
                    if name in server_control_variate and name in client_control_variate:
                        correction = (server_control_variate[name] - client_control_variate[name]).to(param.device)
                        param.grad = param.grad + correction
        
        # Apply differential privacy
        if self.noise_scale > 0:
            param_groups = self.optimizer.param_groups
            
            for group in param_groups:
                # Clip gradients for each parameter group
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(group['params'], self.max_grad_norm)
                
                # Add noise to gradients
                for p in group['params']:
                    if p.requires_grad and p.grad is not None:
                        noise = torch.randn_like(p.grad) * self.noise_scale * self.max_grad_norm
                        p.grad = p.grad + noise
        
        # Apply optimizer step
        self.optimizer.step()
        
    def state_dict(self):
        """Get state dict from optimizer."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict to optimizer."""
        self.optimizer.load_state_dict(state_dict)


# Federated System.

# Normal FedAvg
class FedAVGModel(nn.Module):
    """Client model for FedAVG and related federated learning algorithms."""
    
    def __init__(self, hf_input_dim: int, lf_input_dim: int, lf_output_dim: int, 
                 hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hf_input_dim = hf_input_dim
        self.lf_input_dim = lf_input_dim
        self.lf_output_dim = lf_output_dim
        self.hidden_dim = hidden_dim
        
        # Encoder for high-frequency data
        self.encoder = LSTMEncoder(
            input_dim=hf_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Decoder for low-frequency prediction
        self.decoder = LSTMDecoder(
            input_dim=lf_input_dim,
            hidden_dim=hidden_dim,
            output_dim=lf_output_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, hf_input: torch.Tensor, lf_input: torch.Tensor) -> dict:
        """
        Forward pass through the model.
        Args:
            hf_input: High-frequency input tensor (batch_size, hf_seq_len, hf_input_dim)
            lf_input: Low-frequency input tensor (batch_size, lf_seq_len, lf_input_dim)
        Returns:
            Dictionary containing low-frequency predictions
        """
        # Encode high-frequency input
        encoder_outputs, encoder_hidden = self.encoder(hf_input)
        
        # Decode with low-frequency decoder
        decoder_output = self.decoder(lf_input, encoder_outputs)
        
        # Extract the last prediction (forecast) from sequence
        if decoder_output.size(1) > 1:
            decoder_output = decoder_output[:, -1:, :]
        
        return {
            "lf_pred": decoder_output
        }

# Secured FedAvg
class PrivacyPreservingFedAVGModel(nn.Module):
    
    """Privacy-preserving client model for FedAVG."""
    def __init__(self, hf_input_dim, lf_input_dim, lf_output_dim, hidden_dim, num_layers=2, dropout=0.2, 
                 noise_scale=0.05, clip_bound=1.0):
        super().__init__()
        self.hf_input_dim = hf_input_dim
        self.lf_input_dim = lf_input_dim
        self.lf_output_dim = lf_output_dim
        self.hidden_dim = hidden_dim
        
        # Regular encoder for high-frequency data
        self.base_encoder = LSTMEncoder(
            input_dim=hf_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Wrap with private encoder
        self.encoder = PrivateEncoder(
            original_encoder=self.base_encoder,
            noise_scale=noise_scale,
            clip_bound=clip_bound
        )
        
        # Decoder for low-frequency prediction
        self.decoder = LSTMDecoderWithBahdanauAttention(
            input_dim=lf_input_dim,
            hidden_dim=hidden_dim,
            output_dim=lf_output_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, hf_input, lf_input):
        """
        Forward pass through the model.
        Args:
            hf_input: High-frequency input tensor (batch_size, hf_seq_len, hf_input_dim)
            lf_input: Low-frequency input tensor (batch_size, lf_seq_len, lf_input_dim)
        Returns:
            Dictionary containing low-frequency predictions
        """
        # Encode high-frequency input with privacy
        encoder_outputs, encoder_hidden = self.encoder(hf_input)
        
        # # Decode with low-frequency decoder
        decoder_output = self.decoder(lf_input, encoder_outputs)
         
        # Extract the last prediction (forecast) from sequence
        if decoder_output.size(1) > 1:
            decoder_output = decoder_output[:, -1:, :]
        
        return {
            "lf_pred": decoder_output
        }
    
class PrivacyPreservingScaffoldModel(nn.Module):
    """Privacy-preserving client model for SCAFFOLD."""
    def __init__(self, hf_input_dim, lf_input_dim, lf_output_dim, hidden_dim, num_layers=2, dropout=0.2, 
                 noise_scale=0.05, clip_bound=1.0):
        super().__init__()
        self.hf_input_dim = hf_input_dim
        self.lf_input_dim = lf_input_dim
        self.lf_output_dim = lf_output_dim
        self.hidden_dim = hidden_dim
        
        # Regular encoder for high-frequency data
        self.base_encoder = LSTMEncoder(
            input_dim=hf_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Wrap with private encoder
        self.encoder = PrivateEncoder(
            original_encoder=self.base_encoder,
            noise_scale=noise_scale,
            clip_bound=clip_bound
        )
        
        # Decoder for low-frequency prediction
        self.decoder = LSTMDecoderWithBahdanauAttention(
            input_dim=lf_input_dim,
            hidden_dim=hidden_dim,
            output_dim=lf_output_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, hf_input, lf_input):
        """
        Forward pass through the model.
        Args:
            hf_input: High-frequency input tensor (batch_size, hf_seq_len, hf_input_dim)
            lf_input: Low-frequency input tensor (batch_size, lf_seq_len, lf_input_dim)
        Returns:
            Dictionary containing low-frequency predictions
        """
        # Encode high-frequency input with privacy
        encoder_outputs, encoder_hidden = self.encoder(hf_input)
        
        # Decode with low-frequency decoder
        decoder_output = self.decoder(lf_input, encoder_outputs)
        
        # Extract the last prediction (forecast) from sequence
        if decoder_output.size(1) > 1:
            decoder_output = decoder_output[:, -1:, :]
        
        return {
            "lf_pred": decoder_output
        }
    
    def get_model_structure(self):
        """Get the structure of model parameters for control variates."""
        return {name: param.shape for name, param in self.named_parameters()}