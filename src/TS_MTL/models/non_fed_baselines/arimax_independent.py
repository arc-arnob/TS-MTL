import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
pd.options.mode.chained_assignment = None
np.seterr(all='ignore')

class CustomScaler:
    """Custom scaler class to handle mean/std scaling and inverse transforms."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std if std != 0 else 1.0
        
    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)
        
    def inverse_transform(self, data):
        return data * (self.std + 1e-8) + self.mean

def run_arimax_independent(
    data_base_path="src/TS_MTL/data/air_quality",
    sites=["station-1", "station-2", "station-3", "station-4", "station-5", "station-6"],
    hf_suffix="-hf.csv",
    lf_suffix="-lf.csv",
    features=["PM2.5", "NO2", "PM10"],
    target=["CO"],
    min_date="2014-09-01",
    max_date="2014-11-12",
    lookback_days=32,
    forecast_horizon=16,
    train_ratio=0.8,
    save_plots=False
):
    """
    Run independent ARIMAX models for each site.
    
    Args:
        data_base_path: Base path to data directory
        sites: List of site names
        hf_suffix: High frequency file suffix
        lf_suffix: Low frequency file suffix
        features: List of feature columns (exogenous variables)
        target: List containing target column name
        min_date: Start date for data filtering
        max_date: End date for data filtering
        lookback_days: Number of historical observations for model
        forecast_horizon: Number of steps to forecast ahead
        train_ratio: Ratio of data for training
        save_plots: Whether to save forecast plots
    
    Returns:
        Dictionary with results for each site and overall metrics
    """
    
    min_date = pd.to_datetime(min_date)
    max_date = pd.to_datetime(max_date)
    target_col = target[0]  # Extract target column name
    exog_cols = features
    
    print(f"Running Independent ARIMAX for {len(sites)} sites")
    print(f"Target: {target_col}, Features: {exog_cols}")
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    
    results = []
    all_predictions = []
    all_actuals = []
    
    for site in sites:
        print(f"\nProcessing site {site}")
        
        # Load data files
        lf_file = os.path.join(data_base_path, f"{site}{lf_suffix}")
        hf_file = os.path.join(data_base_path, f"{site}{hf_suffix}")
        
        if not os.path.exists(lf_file) or not os.path.exists(hf_file):
            print(f"Missing data files for {site}. Skipping.")
            continue
        
        lf_data = pd.read_csv(lf_file)
        hf_data = pd.read_csv(hf_file)
        
        # Standardize column names
        lf_data.rename(columns={"time": "Time"}, inplace=True)
        hf_data.rename(columns={"time": "Time"}, inplace=True)
        
        # Convert time columns
        lf_data['Time'] = pd.to_datetime(lf_data['Time'])
        hf_data['Time'] = pd.to_datetime(hf_data['Time'])
        
        # Filter by date range
        lf_data = lf_data[(lf_data['Time'] >= min_date) & (lf_data['Time'] <= max_date)]
        hf_data = hf_data[(hf_data['Time'] >= min_date) & (hf_data['Time'] <= max_date)]
        
        if len(lf_data) < 30 or len(hf_data) < 30:
            print(f"Insufficient data for {site}. Skipping.")
            continue
        
        # Merge LF and HF data
        hf_hourly = hf_data.groupby(hf_data['Time'].dt.floor('H')).mean().reset_index()
        merged_data = pd.merge(lf_data, hf_hourly, on='Time', how='inner', suffixes=('', '_hf'))
        merged_data = merged_data.sort_values('Time').reset_index(drop=True)
        
        if len(merged_data) < 30:
            print(f"Insufficient merged data for {site}. Skipping.")
            continue
        
        # Check if required columns exist
        missing_cols = [col for col in [target_col] + exog_cols if col not in merged_data.columns]
        if missing_cols:
            print(f"Missing columns {missing_cols} for {site}. Skipping.")
            continue
        
        # Check stationarity
        adf_result = adfuller(merged_data[target_col].dropna())
        diff_order = 1 if adf_result[1] > 0.05 else 0
        
        # Train/test split
        train_size = int(len(merged_data) * train_ratio)
        train_data = merged_data.iloc[:train_size]
        test_data = merged_data.iloc[train_size:]
        
        if len(test_data) < forecast_horizon:
            print(f"Insufficient test data for {site}. Skipping.")
            continue
        
        # Scaling based on training data
        target_scaler = CustomScaler(train_data[target_col].mean(), train_data[target_col].std())
        exog_mean = train_data[exog_cols].mean()
        exog_std = train_data[exog_cols].std().replace(0, 1.0)
        
        # Scale the entire dataset
        merged_data_scaled = merged_data.copy()
        merged_data_scaled[target_col] = target_scaler.transform(merged_data[target_col])
        for col in exog_cols:
            merged_data_scaled[col] = (merged_data[col] - exog_mean[col]) / (exog_std[col] + 1e-8)
        
        # Rolling window forecasting
        site_predictions = []
        site_actuals = []
        
        for i in range(0, len(test_data) - forecast_horizon + 1, forecast_horizon):
            history_end_idx = train_size + i
            history_start_idx = max(0, history_end_idx - lookback_days)
            
            history_scaled = merged_data_scaled.iloc[history_start_idx:history_end_idx]
            forecast_end_idx = history_end_idx + forecast_horizon
            
            if forecast_end_idx > len(merged_data_scaled):
                break
                
            forecast_period = merged_data_scaled.iloc[history_end_idx:forecast_end_idx]
            
            try:
                # Fit ARIMAX model
                p = min(lookback_days, 33)
                model = SARIMAX(
                    history_scaled[target_col].values,
                    exog=history_scaled[exog_cols].values,
                    order=(p, diff_order, 1),
                    enforce_stationarity=False
                )
                model_fit = model.fit(disp=False)
                
                # Forecast
                scaled_pred = model_fit.forecast(steps=forecast_horizon, exog=forecast_period[exog_cols].values)
                
                site_predictions.extend(scaled_pred)
                site_actuals.extend(forecast_period[target_col].values)
                
            except Exception as e:
                print(f"Error in forecasting window {i} for {site}: {str(e)}")
                continue
        
        # Calculate metrics for this site
        if len(site_predictions) > 0:
            mae = mean_absolute_error(site_actuals, site_predictions)
            mse = mean_squared_error(site_actuals, site_predictions)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE
            non_zero_idx = np.where(np.array(site_actuals) != 0)[0]
            if len(non_zero_idx) > 0:
                mape = np.mean(np.abs((np.array(site_actuals)[non_zero_idx] - np.array(site_predictions)[non_zero_idx]) / 
                                      np.array(site_actuals)[non_zero_idx])) * 100
            else:
                mape = np.nan
            
            print(f"Site {site} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
            
            results.append({
                'site': site,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'predictions': len(site_predictions)
            })
            
            all_predictions.extend(site_predictions)
            all_actuals.extend(site_actuals)
            
            # Save plot if requested
            if save_plots:
                plt.figure(figsize=(12, 6))
                plt.plot(site_actuals, label='Actual')
                plt.plot(site_predictions, label='Predicted', color='red')
                plt.title(f'Site {site} - Independent ARIMAX Forecast')
                plt.xlabel('Time Steps')
                plt.ylabel(target_col)
                plt.legend()
                plt.savefig(f'arimax_independent_{site}.png')
                plt.close()
        else:
            print(f"No valid predictions for {site}")
    
    # Calculate overall metrics
    if len(all_predictions) > 0:
        overall_mae = mean_absolute_error(all_actuals, all_predictions)
        overall_mse = mean_squared_error(all_actuals, all_predictions)
        overall_rmse = np.sqrt(overall_mse)
        
        non_zero_idx = np.where(np.array(all_actuals) != 0)[0]
        if len(non_zero_idx) > 0:
            overall_mape = np.mean(np.abs((np.array(all_actuals)[non_zero_idx] - np.array(all_predictions)[non_zero_idx]) / 
                                          np.array(all_actuals)[non_zero_idx])) * 100
        else:
            overall_mape = np.nan
        
        print(f"\nOverall Independent ARIMAX Results:")
        print(f"MAE: {overall_mae:.4f}")
        print(f"MSE: {overall_mse:.4f}")
        print(f"RMSE: {overall_rmse:.4f}")
        print(f"MAPE: {overall_mape:.2f}%")
        
        # Save predictions and targets to log file
        output_file = 'preds_targets_log.txt'
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        preds_list = [float(x) for x in all_predictions]
        targs_list = [float(x) for x in all_actuals]
        
        with open(output_file, 'a') as f:
            f.write(f"# Model: Independent ARIMAX, Sites: {len(results)}, Horizon: {forecast_horizon}\n")
            f.write("predictions: " + json.dumps(preds_list) + "\n")
            f.write("targets:     " + json.dumps(targs_list) + "\n\n")
        
        return {
            'overall_metrics': {
                'mae': overall_mae,
                'mse': overall_mse,
                'rmse': overall_rmse,
                'mape': overall_mape
            },
            'site_results': results,
            'predictions': preds_list,
            'targets': targs_list
        }
    else:
        print("No valid predictions generated across all sites")
        return None


