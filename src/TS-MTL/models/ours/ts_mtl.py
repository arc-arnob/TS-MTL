import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional
import time
import json
import dateutil
from datetime import datetime
from collections import defaultdict

#####################################################
# Hard Parameter Sharing Model
#####################################################

class HardParameterSharingModel(nn.Module):
    """
    Mixed frequency model with hard parameter sharing.
    Only the final output layers are client-specific.
    """
    def __init__(self, hf_input_dim, lf_input_dim, lf_output_dim, 
                 hidden_dim, client_ids=None, num_layers=2, dropout=0.1):
        super().__init__()
        self.hf_input_dim = hf_input_dim
        self.lf_input_dim = lf_input_dim
        self.lf_output_dim = lf_output_dim
        self.hidden_dim = hidden_dim
        self.client_ids = client_ids or []
        
        # Shared encoder - LSTM based
        self.shared_encoder = nn.LSTM(
            input_size=hf_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Shared decoder - LSTM based
        self.shared_decoder = nn.LSTM(
            input_size=lf_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Shared intermediate processing layer
        self.shared_intermediate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Client-specific final output layers
        if client_ids:
            self.client_output_layers = nn.ModuleDict({
                str(client_id): nn.Linear(hidden_dim, lf_output_dim)
                for client_id in client_ids
            })
        
        # Shared output layer (used for unknown clients or when no client_ids provided)
        self.shared_output_layer = nn.Linear(hidden_dim, lf_output_dim)
    
    def get_shared_parameters(self):
        """Get all shared parameters of the model"""
        shared_params = []
        for name, param in self.named_parameters():
            if not name.startswith('client_output_layers'):
                shared_params.append(param)
        return shared_params
    
    def get_client_parameters(self, client_id):
        """Get parameters specific to a client"""
        client_id_str = str(client_id)
        if hasattr(self, 'client_output_layers') and client_id_str in self.client_output_layers:
            return list(self.client_output_layers[client_id_str].parameters())
        return []
    
    def apply_attention(self, decoder_output, encoder_outputs):
        """Apply attention mechanism between decoder and encoder outputs."""
        batch_size, seq_len, _ = decoder_output.size()
        enc_seq_len = encoder_outputs.size(1)
        
        # Expand decoder output for attention calculation
        decoder_output_expanded = decoder_output.unsqueeze(2).expand(
            batch_size, seq_len, enc_seq_len, self.hidden_dim
        )
        
        # Expand encoder outputs for attention calculation
        encoder_outputs_expanded = encoder_outputs.unsqueeze(1).expand(
            batch_size, seq_len, enc_seq_len, self.hidden_dim
        )
        
        # Concatenate for attention calculation
        combined = torch.cat([decoder_output_expanded, encoder_outputs_expanded], dim=3)
        
        # Calculate attention scores
        attention_scores = self.attention(combined).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=2)
        
        # Apply attention weights to encoder outputs
        context_vectors = torch.bmm(
            attention_weights.view(batch_size * seq_len, 1, enc_seq_len),
            encoder_outputs.repeat(seq_len, 1, 1)
        ).view(batch_size, seq_len, self.hidden_dim)
        
        return context_vectors
    
    def forward(self, hf_input, lf_input, client_id=None):
        """
        Forward pass through the model.
        Args:
            hf_input: High-frequency input tensor (batch_size, hf_seq_len, hf_input_dim)
            lf_input: Low-frequency input tensor (batch_size, lf_seq_len, lf_input_dim)
            client_id: Optional client identifier for client-specific processing
        Returns:
            Dictionary containing low-frequency predictions
        """
        batch_size = hf_input.size(0)
        
        # Step 1: Encode high-frequency input
        encoder_outputs, _ = self.shared_encoder(hf_input)
        
        # Step 2: Decode with shared decoder
        decoder_outputs, _ = self.shared_decoder(lf_input)
        
        # Step 3: Apply attention between decoder and encoder
        context_vectors = self.apply_attention(decoder_outputs, encoder_outputs) #BUG 
        
        # Step 4: Concatenate decoder output with context
        last_decoder_output = decoder_outputs[:, -1:, :]
        last_context = context_vectors[:, -1:, :]
        combined = torch.cat([last_decoder_output, last_context], dim=2)
        
        # Step 5: Apply shared intermediate processing
        processed = self.shared_intermediate(combined)
        
        # Step 6: Apply client-specific or shared output layer
        if client_id is not None and hasattr(self, 'client_output_layers'):
            # Group samples by client ID for efficient processing
            if isinstance(client_id, torch.Tensor):
                if client_id.numel() == 1:  # Single client ID for all samples
                    client_id_str = str(client_id.item())
                    if client_id_str in self.client_output_layers:
                        lf_pred = self.client_output_layers[client_id_str](processed)
                    else:
                        lf_pred = self.shared_output_layer(processed)
                else:  # Batch of client IDs
                    # Initialize output with same shape as processed
                    lf_pred = torch.zeros(batch_size, 1, self.lf_output_dim).to(hf_input.device)
                    
                    # Process each unique client ID in the batch together
                    unique_clients = torch.unique(client_id)
                    for c in unique_clients:
                        c_str = str(c.item())
                        client_indices = (client_id == c).nonzero(as_tuple=True)[0]
                        
                        if c_str in self.client_output_layers:
                            # Apply client-specific layer
                            client_output = self.client_output_layers[c_str](processed[client_indices])
                            lf_pred[client_indices] = client_output
                        else:
                            # Use shared output layer
                            shared_output = self.shared_output_layer(processed[client_indices])
                            lf_pred[client_indices] = shared_output
            else:
                # Single client ID as string or int
                client_id_str = str(client_id)
                if client_id_str in self.client_output_layers:
                    lf_pred = self.client_output_layers[client_id_str](processed)
                else:
                    lf_pred = self.shared_output_layer(processed)
        else:
            # Use shared output layer
            lf_pred = self.shared_output_layer(processed)
        
        return {"lf_pred": lf_pred}
