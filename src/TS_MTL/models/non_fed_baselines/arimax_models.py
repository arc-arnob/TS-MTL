"""
ARIMAX model classes for registry compatibility.
These are lightweight wrappers since ARIMAX models are statistical models
that don't follow the same PyTorch pattern as neural networks.
"""


class ARIMAXBaseModel:
    """Base class for ARIMAX models - minimal interface for CLI compatibility."""
    
    def __init__(self, **kwargs):
        # ARIMAX models don't need most neural network parameters
        # but we accept them for compatibility
        self.model_type = "arimax"
        
    def to(self, device):
        # ARIMAX models don't use GPU
        return self
        
    def train(self):
        # ARIMAX training is handled differently
        pass
        
    def eval(self):
        # ARIMAX evaluation is handled differently  
        pass


class ARIMAXIndependentModel(ARIMAXBaseModel):
    """Independent ARIMAX model (one per site)."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_variant = "independent"


class ARIMAXGlobalModel(ARIMAXBaseModel):
    """Global ARIMAX model (pooled across sites)."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_variant = "global"
