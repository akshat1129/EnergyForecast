import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def load_and_process_data(filepath):
    print(f"Loading data from {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, sep=';', 
                     parse_dates={'dt': ['Date', 'Time']}, 
                     infer_datetime_format=True, 
                     dayfirst=True,
                     low_memory=False, 
                     na_values=['nan', '?'])

    df = df.set_index('dt')
    df = df.fillna(method='ffill')

    cols_to_numeric = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Feature Engineering
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Resample to Hourly
    df_hourly = df.resample('H').mean()
    df_hourly = df_hourly.dropna()
    
    print(f"Data processed. Hourly shape: {df_hourly.shape}")
    return df_hourly

# ==========================================
# 2. IDEAL MODEL RUNNER
# ==========================================

def run_ideal_model(data, n_lag):
    # Configuration
    features_to_use = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
                       'Sub_metering_3', 'hour', 'day_of_week']
    
    print(f"\nRunning Ideal Model: LightGBM, Lag={n_lag} hours")
    
    # Select features
    dataset = data[features_to_use].values
    
    # Frame as supervised learning
    reframed = series_to_supervised(dataset, n_in=n_lag, n_out=1)
    
    # Drop columns we don't want to predict (keep only var1(t) which is Global_active_power)
    n_features = len(features_to_use)
    if n_features > 1:
        reframed.drop(reframed.columns[list(range(reframed.shape[1] - (n_features - 1), reframed.shape[1]))], axis=1, inplace=True)
        
    values = reframed.values
    
    # Split Train/Test (Last week)
    n_test = 168
    train = values[:-n_test, :]
    test = values[-n_test:, :]
    
    # Split X, y
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    
    # Train LightGBM
    print(f"Training LightGBM (Lag {n_lag})...")
    start_time = time.time()
    model = lgb.LGBMRegressor(n_estimators=1000, n_jobs=-1, random_state=42)
    model.fit(train_X, train_y)
    
    predictions = model.predict(test_X)
    inference_time = time.time() - start_time
    
    mae, rmse = evaluate_forecast(test_y, predictions)
    print(f"  -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, Time: {inference_time:.4f}s")
    
    return test_y, predictions, mae, rmse

def main():
    dataset_path = r"c:/Users/Akshat/Desktop/AISmartHome/individual+household+electric+power+consumption/household_power_consumption.txt"
    try:
        data = load_and_process_data(dataset_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Lags to test: 24h, 3 Days (72h), 7 Days (168h)
    lags = [24, 72, 168]
    results = {}
    
    plt.figure(figsize=(14, 8))
    
    # We need to capture the actual values from one of the runs (they are the same test set)
    actual_values = None
    
    colors = {24: 'blue', 72: 'green', 168: 'orange'}
    
    for lag in lags:
        actual, predicted, mae, rmse = run_ideal_model(data, lag)
        results[lag] = {'mae': mae, 'rmse': rmse, 'predicted': predicted}
        
        if actual_values is None:
            actual_values = actual
            plt.plot(actual_values, label='Actual (Last Week)', color='black', linewidth=2)
            
        plt.plot(predicted, label=f'Lag {lag}h (MAE={mae:.3f})', color=colors[lag], linestyle='--')

    plt.title('Ideal Energy Forecast Comparison (LightGBM: 24h vs 72h vs 168h Lag)')
    plt.xlabel('Hour')
    plt.ylabel('Global Active Power (kW)')
    plt.legend()
    plt.grid(True)
    plt.savefig('ideal_forecast_comparison.png')
    print("\nForecast plot saved to 'ideal_forecast_comparison.png'")
    
    # Print Summary
    print("\n" + "="*40)
    print(f"{'Lag':<10} | {'MAE':<10} | {'RMSE':<10}")
    print("-" * 40)
    for lag in lags:
        print(f"{lag:<10} | {results[lag]['mae']:<10.4f} | {results[lag]['rmse']:<10.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
