class CustomScaler:
    """Custom scaler class to handle mean/std scaling and inverse transforms."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)
        
    def inverse_transform(self, data):
        return data * (self.std + 1e-8) + self.mean