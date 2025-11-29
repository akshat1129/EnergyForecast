import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

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
# 2. MODEL TRAINING FUNCTIONS
# ==========================================

def train_linear_regression(train_X, train_y, test_X, test_y):
    print("Training Linear Regression...")
    start_time = time.time()
    
    model = LinearRegression()
    model.fit(train_X, train_y)
    predictions = model.predict(test_X)
    
    inference_time = time.time() - start_time
    mae, rmse, r2 = evaluate_forecast(test_y, predictions)
    return predictions, mae, rmse, r2, inference_time

def train_xgboost(train_X, train_y, test_X, test_y):
    print("Training XGBoost...")
    start_time = time.time()
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, n_jobs=-1)
    model.fit(train_X, train_y)
    predictions = model.predict(test_X)
    
    inference_time = time.time() - start_time
    mae, rmse, r2 = evaluate_forecast(test_y, predictions)
    return predictions, mae, rmse, r2, inference_time

def train_lightgbm(train_X, train_y, test_X, test_y):
    print("Training LightGBM...")
    start_time = time.time()
    
    model = lgb.LGBMRegressor(n_estimators=500, n_jobs=-1)
    model.fit(train_X, train_y)
    predictions = model.predict(test_X)
    
    inference_time = time.time() - start_time
    mae, rmse, r2 = evaluate_forecast(test_y, predictions)
    return predictions, mae, rmse, r2, inference_time

def train_lstm(train_X, train_y, test_X, test_y):
    print("Training LSTM...")
    start_time = time.time()
    
    # Reshape for LSTM [samples, timesteps, features]
    train_X_reshaped = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X_reshaped = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X_reshaped.shape[1], train_X_reshaped.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    # Reduced epochs for efficiency in this demo, increase for production
    model.fit(train_X_reshaped, train_y, epochs=10, batch_size=72, verbose=0, shuffle=False)
    
    predictions = model.predict(test_X_reshaped).flatten()
    
    inference_time = time.time() - start_time
    mae, rmse, r2 = evaluate_forecast(test_y, predictions)
    return predictions, mae, rmse, r2, inference_time

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    dataset_path = r"c:/Users/Akshat/Desktop/AISmartHome/individual+household+electric+power+consumption/household_power_consumption.txt"
    try:
        data = load_and_process_data(dataset_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Configuration: Full Features & 3-Day Lag
    features_to_use = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
                       'Sub_metering_3', 'hour', 'day_of_week']
    n_lag = 72
    
    print(f"\nConfiguration: Lag={n_lag} hours (3 Days), Features={len(features_to_use)}")

    # Prepare Data
    dataset = data[features_to_use].values
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
    
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    
    results = {}
    
    # --- Train Models ---
    
    # 1. Linear Regression
    try:
        pred, mae, rmse, r2, time_taken = train_linear_regression(train_X, train_y, test_X, test_y)
        results['Linear Regression'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Time': time_taken, 'Pred': pred}
    except Exception as e:
        print(f"Linear Regression failed: {e}")

    # 2. XGBoost
    try:
        pred, mae, rmse, r2, time_taken = train_xgboost(train_X, train_y, test_X, test_y)
        results['XGBoost'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Time': time_taken, 'Pred': pred}
    except Exception as e:
        print(f"XGBoost failed: {e}")

    # 3. LSTM
    try:
        pred, mae, rmse, r2, time_taken = train_lstm(train_X, train_y, test_X, test_y)
        results['LSTM'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Time': time_taken, 'Pred': pred}
    except Exception as e:
        print(f"LSTM failed: {e}")

    # 4. LightGBM
    try:
        pred, mae, rmse, r2, time_taken = train_lightgbm(train_X, train_y, test_X, test_y)
        results['LightGBM'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Time': time_taken, 'Pred': pred}
    except Exception as e:
        print(f"LightGBM failed: {e}")
        
    # --- Print Results ---
    print("\n" + "="*80)
    print(f"{'Model':<20} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10} | {'Time (s)':<10}")
    print("-" * 80)
    for name, metrics in results.items():
        print(f"{name:<20} | {metrics['MAE']:<10.4f} | {metrics['RMSE']:<10.4f} | {metrics['R2']:<10.4f} | {metrics['Time']:<10.4f}")
    print("="*80)
    
    # --- Visualize ---
    plt.figure(figsize=(14, 8))
    plt.plot(test_y[:168], label='Actual', color='black', linewidth=2)
    
    colors = {'Linear Regression': 'blue', 'XGBoost': 'red', 'LSTM': 'purple', 'LightGBM': 'orange'}
    
    for name, metrics in results.items():
        plt.plot(metrics['Pred'][:168], label=f"{name} (MAE={metrics['MAE']:.3f})", 
                 color=colors.get(name, 'gray'), linestyle='--')
            
    plt.title('Energy Forecast Comparison (3-Day Lag, Full Features)')
    plt.xlabel('Hour')
    plt.ylabel('Global Active Power (kW)')
    plt.legend()
    plt.grid(True)
    plt.savefig('energy_forecast.png')
    print("\nComparison plot saved to 'energy_forecast.png'")

if __name__ == "__main__":
    main()
