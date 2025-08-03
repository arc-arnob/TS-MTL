"""
Federated Learning Module

This module contains implementations of various federated learning algorithms
including FedAvg, Personalized FedAvg, and other baselines.
"""

from .base_federated import BaseFederatedSystem
from .fed_avg import FederatedAVGSystem
from .per_fed_avg import PersonalizedFedAVGSystem, SimplePFedAvgSystem
from .model_components import FedAVGModel, LSTMEncoder, LSTMDecoder, TemporalAttention

__all__ = [
    'BaseFederatedSystem',
    'FederatedAVGSystem', 
    'PersonalizedFedAVGSystem',
    'SimplePFedAvgSystem',
    'FedAVGModel',
    'LSTMEncoder',
    'LSTMDecoder', 
    'TemporalAttention'
]
