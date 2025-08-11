# Multi-frequency data preprocessors for TS-MTL project
# 
# This package contains data preprocessing modules for converting raw datasets
# into mixed-frequency format suitable for multi-task learning experiments.
#
# Available preprocessors:
# - air_quality_data_preprocessor: Beijing air quality data (6 stations)
# - load_preprocessor: Electricity load forecasting data (20 zones)
# - wind_preprocessor: Wind power forecasting data (7 farms)
# - spain_data_preprocessor: Spain multi-site load data (multiple cities)

__version__ = "1.0.0"
__author__ = "TS-MTL Team"

# Import main preprocessing functions for convenience
try:
    from .air_quality_data_preprocessor import process_airquality_data
    from .load_preprocessor import process_load_forecasting_data
    from .wind_preprocessor import process_wind_farm_data
    from .spain_data_preprocessor import process_spain_multisite_data
    
    __all__ = [
        'process_airquality_data',
        'process_load_forecasting_data', 
        'process_wind_farm_data',
        'process_spain_multisite_data'
    ]
except ImportError:
    # Handle cases where dependencies might not be available
    __all__ = []
