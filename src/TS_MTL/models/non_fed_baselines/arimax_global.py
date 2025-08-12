import os
import warnings
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
pd.options.mode.chained_assignment = None
np.seterr(all='ignore')


class CustomScaler:
    """Custom scaler class to handle mean/std scaling."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std if std != 0 else 1.0

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)


def run_arimax_global(
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
    Run global ARIMAX model with all sites pooled together.
    
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
        Dictionary with global results and per-site metrics
    """
    
    min_date = pd.to_datetime(min_date)
    max_date = pd.to_datetime(max_date)
    target_col = target[0]  # Extract target column name
    exog_cols = features
    
    print(f"Running Global ARIMAX with {len(sites)} sites pooled")
    print(f"Target: {target_col}, Features: {exog_cols}")
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    
    # 1) Read & merge each site's LF+HF, collect into dfs
    dfs = []
    valid_sites = []
    
    for site in sites:
        lf_file = os.path.join(data_base_path, f"{site}{lf_suffix}")
        hf_file = os.path.join(data_base_path, f"{site}{hf_suffix}")
        
        if not os.path.exists(lf_file) or not os.path.exists(hf_file):
            print(f"Missing data files for {site}. Skipping.")
            continue
        
        lf = pd.read_csv(lf_file)
        hf = pd.read_csv(hf_file)
        
        # Standardize column names
        lf.rename(columns={"time": "Time"}, inplace=True)
        hf.rename(columns={"time": "Time"}, inplace=True)
        
        lf["Time"] = pd.to_datetime(lf["Time"])
        hf["Time"] = pd.to_datetime(hf["Time"])
        
        # Filter by date range
        lf = lf[(lf.Time >= min_date) & (lf.Time <= max_date)]
        hf = hf[(hf.Time >= min_date) & (hf.Time <= max_date)]
        
        if len(lf) < 30 or len(hf) < 30:
            print(f"Insufficient data for {site}. Skipping.")
            continue
        
        # Merge LF and HF data
        hf_hourly = hf.groupby(hf.Time.dt.floor("H")).mean().reset_index()
        merged = pd.merge(lf, hf_hourly, on="Time", how="inner", suffixes=("", "_hf"))
        
        if len(merged) < 30:
            print(f"Insufficient merged data for {site}. Skipping.")
            continue
        
        # Check if required columns exist
        required_cols = [target_col] + exog_cols
        missing_cols = [col for col in required_cols if col not in merged.columns]
        if missing_cols:
            print(f"Missing columns {missing_cols} for {site}. Skipping.")
            continue
        
        # Keep only required columns and add site identifier
        merged = merged[["Time"] + required_cols].copy()
        merged["site_id"] = site
        dfs.append(merged)
        valid_sites.append(site)
        print(f"Added {site} with {len(merged)} records")

    if not dfs:
        print("No valid data found for any site")
        return None

    # 2) Pool into global_df
    global_df = pd.concat(dfs, ignore_index=True)
    global_df.sort_values(["Time", "site_id"], inplace=True)
    global_df.reset_index(drop=True, inplace=True)
    
    print(f"Global dataset: {len(global_df)} records from {len(valid_sites)} sites")

    # 3) Site dummies as exogenous variables
    site_dummies = pd.get_dummies(global_df["site_id"], prefix="site")
    global_df = pd.concat([global_df, site_dummies], axis=1)

    # Updated exogenous columns to include site dummies
    exog_cols_with_sites = exog_cols + list(site_dummies.columns)

    # 4) Check stationarity and determine differencing order
    adf_p = adfuller(global_df[target_col].ffill())[1]
    d = 1 if adf_p > 0.05 else 0
    print(f"ADF p-value = {adf_p:.4f} → differencing order d = {d}")

    # 5) Time‐based train/test split
    cutoff_time = global_df["Time"].quantile(train_ratio)
    train_df = global_df[global_df.Time <= cutoff_time].copy()
    test_df = global_df[global_df.Time > cutoff_time].copy()
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # 6) Scale based on training data
    target_scaler = CustomScaler(train_df[target_col].mean(), train_df[target_col].std())
    exog_mean = train_df[exog_cols_with_sites].mean()
    exog_std = train_df[exog_cols_with_sites].std().replace(0, 1.0)
    
    # Apply scaling to all dataframes
    for df in (train_df, test_df, global_df):
        df[target_col] = target_scaler.transform(df[target_col])
        # Scale exogenous columns individually to avoid DataFrame assignment issues
        for col in exog_cols_with_sites:
            df[col] = (df[col] - exog_mean[col]) / (exog_std[col] + 1e-8)

    # 7) Rolling‐window ARIMAX forecasting
    all_preds, all_actuals, all_sites = [], [], []
    total_len = len(global_df)
    train_end_idx = len(train_df)
    
    for i in range(0, len(test_df), forecast_horizon):
        hist_end = train_end_idx + i
        hist_start = max(0, hist_end - lookback_days)
        
        if hist_end + forecast_horizon > total_len:
            break

        hist = global_df.iloc[hist_start:hist_end]
        fut = global_df.iloc[hist_end:hist_end + forecast_horizon]

        try:
            model = SARIMAX(
                hist[target_col],
                exog=hist[exog_cols_with_sites],
                order=(min(lookback_days, 33), d, 1),
                enforce_stationarity=False
            ).fit(disp=False)

            pred = model.forecast(
                steps=forecast_horizon,
                exog=fut[exog_cols_with_sites]
            )
            
            all_preds.extend(pred)
            all_actuals.extend(fut[target_col].values)
            all_sites.extend(fut["site_id"].values)
            
        except Exception as e:
            print(f"Error in forecasting window {i}: {str(e)}")
            continue

    # 8) Calculate metrics
    if not all_preds:
        print("No valid predictions generated")
        return None
        
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    all_sites = np.array(all_sites)
    
    # Overall metrics
    overall_mae = mean_absolute_error(all_actuals, all_preds)
    overall_mse = mean_squared_error(all_actuals, all_preds)
    overall_rmse = np.sqrt(overall_mse)
    
    # Calculate MAPE
    non_zero_idx = np.where(all_actuals != 0)[0]
    if len(non_zero_idx) > 0:
        overall_mape = np.mean(np.abs((all_actuals[non_zero_idx] - all_preds[non_zero_idx]) / 
                                      all_actuals[non_zero_idx])) * 100
    else:
        overall_mape = np.nan
    
    print(f"\nGlobal ARIMAX Results:")
    print(f"MAE: {overall_mae:.4f}")
    print(f"MSE: {overall_mse:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"MAPE: {overall_mape:.2f}%")
    
    # Per-site metrics
    dfm = pd.DataFrame({
        "site_id": all_sites,
        "pred": all_preds,
        "actual": all_actuals
    })
    
    per_site_mae = dfm.groupby("site_id").apply(
        lambda g: mean_absolute_error(g["actual"], g["pred"])
    )
    
    print("\nMAE by site:")
    for site, mae in per_site_mae.items():
        print(f"  {site}: {mae:.4f}")
    print(f"\nAverage MAE across sites: {per_site_mae.mean():.4f}")
    
    # Save predictions and targets to log file
    output_file = 'preds_targets_log.txt'
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    preds_list = [float(x) for x in all_preds]
    targs_list = [float(x) for x in all_actuals]
    
    with open(output_file, 'a') as f:
        f.write(f"# Model: Global ARIMAX, Sites: {len(valid_sites)}, Horizon: {forecast_horizon}\n")
        f.write("predictions: " + json.dumps(preds_list) + "\n")
        f.write("targets:     " + json.dumps(targs_list) + "\n\n")
    
    # Save plot if requested
    if save_plots:
        plt.figure(figsize=(12, 5))
        plt.plot(all_actuals, label="Actual (scaled)")
        plt.plot(all_preds, "--", label="Predicted (scaled)")
        plt.legend()
        plt.title(f"Global ARIMAX Forecast (scaled)\nAvg MAE: {per_site_mae.mean():.4f}")
        plt.xlabel("Time Steps")
        plt.ylabel(f"{target_col} (scaled)")
        plt.tight_layout()
        plt.savefig("global_arimax_forecast_scaled.png")
        print("Saved plot to global_arimax_forecast_scaled.png")
    
    return {
        'overall_metrics': {
            'mae': overall_mae,
            'mse': overall_mse,
            'rmse': overall_rmse,
            'mape': overall_mape
        },
        'per_site_mae': per_site_mae.to_dict(),
        'avg_site_mae': per_site_mae.mean(),
        'valid_sites': valid_sites,
        'predictions': preds_list,
        'targets': targs_list
    }



