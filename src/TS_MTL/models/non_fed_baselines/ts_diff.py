import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#####################################################
# TSDiff Model Components
#####################################################

# For internal use only
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# For internal use only
class ResidualBlock(nn.Module):
    """Residual block for TSDiff."""
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels)
        )
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.activation = nn.SiLU()
        
    def forward(self, x, time_emb):
        # x: (batch, channels, length)
        # time_emb: (batch, time_emb_dim)
        
        residual = x
        
        # First conv block
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        
        # Time embedding
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb[:, :, None]  # (batch, channels, 1)
        x = x + time_emb
        
        # Second conv block
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        
        return x + residual


# External use
class TSDiffModel(nn.Module):
    """TSDiff model for time series diffusion."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, time_emb_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_embeddings = SinusoidalPositionEmbeddings(time_emb_dim)
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_emb_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
    
    
    
    def forward(self, x, t):
        # x: (batch, length, features)
        # t: (batch,) timestep
        
        # Transpose for conv1d: (batch, features, length)
        x = x.transpose(1, 2)
        
        # Time embedding
        time_emb = self.time_embeddings(t)
        
        # Input projection
        x = self.input_proj(x)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x, time_emb)
        
        # Output projection
        x = self.output_proj(x)
        
        # Transpose back: (batch, length, features)
        x = x.transpose(1, 2)
        
        return x


#####################################################
# Diffusion Process
#####################################################

# External use
class DiffusionProcess:
    """Handles the forward and reverse diffusion process."""
    def __init__(self, num_timesteps=100, beta_start=1e-4, beta_end=0.1):
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def to(self, device):
        """Move tensors to device."""
        attrs = ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'posterior_variance']
        for attr in attrs:
            setattr(self, attr, getattr(self, attr).to(device))
        return self
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x, t, condition=None, guidance_scale=1.0):
        """Single step of reverse diffusion."""
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])
        
        # Reshape for broadcasting
        betas_t = betas_t[:, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None]
        sqrt_recip_alphas_t = sqrt_recip_alphas_t[:, None, None]
        
        # Model prediction
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        # Apply guidance if condition is provided
        if condition is not None and guidance_scale > 1.0:
            # Implement observation self-guidance
            model_mean = self.apply_guidance(model_mean, condition, guidance_scale)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def apply_guidance(self, x, condition, guidance_scale):
        """Apply observation self-guidance."""
        obs_indices, obs_values = condition
        
        # Create guidance based on MSE loss
        x_guided = x.clone()
        if obs_indices is not None and len(obs_indices) > 0:
            # Simple guidance: blend predicted values with observations
            for i, (idx, val) in enumerate(zip(obs_indices, obs_values)):
                if idx < x.shape[1]:  # Make sure index is valid
                    guidance = (val - x_guided[:, idx]) * guidance_scale * 0.1
                    x_guided[:, idx] += guidance
        
        return x_guided
    
    def p_sample_loop(self, model, shape, condition=None, guidance_scale=1.0, device='cpu'):
        """Full reverse diffusion sampling loop."""
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, condition, guidance_scale)
            
        return x