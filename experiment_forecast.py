import pandas as pd
import numpy as np
import os
import time
import math

# Machine Learning Libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# Deep Learning Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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
    
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
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
        
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    df_hourly = df.resample('H').mean()
    df_hourly = df_hourly.dropna()
    
    print(f"Data processed. Hourly shape: {df_hourly.shape}")
    return df_hourly

# ==========================================
# 2. EXPERIMENT RUNNER
# ==========================================

def run_experiment(data, n_lag, features_to_use, model_type='xgboost'):
    print(f"\nRunning Experiment: Model={model_type}, Lag={n_lag}")
    
    if 'Global_active_power' not in features_to_use:
        features_to_use = ['Global_active_power'] + features_to_use
        
    dataset = data[features_to_use].values
    reframed = series_to_supervised(dataset, n_in=n_lag, n_out=1)
    
    n_features = len(features_to_use)
    if n_features > 1:
        reframed.drop(reframed.columns[list(range(reframed.shape[1] - (n_features - 1), reframed.shape[1]))], axis=1, inplace=True)
        
    values = reframed.values
    
    n_test = 168
    train = values[:-n_test, :]
    test = values[-n_test:, :]
    
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    
    start_time = time.time()
    predictions = []
    
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, n_jobs=-1)
        model.fit(train_X, train_y)
        predictions = model.predict(test_X)
        
    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor(n_estimators=500, n_jobs=-1)
        model.fit(train_X, train_y)
        predictions = model.predict(test_X)
        
    elif model_type == 'random_forest':
        # Reduced estimators for speed as per previous findings
        model = RandomForestRegressor(n_estimators=20, n_jobs=-1, random_state=42)
        model.fit(train_X, train_y)
        predictions = model.predict(test_X)
        
    elif model_type == 'linear_regression':
        model = LinearRegression()
        model.fit(train_X, train_y)
        predictions = model.predict(test_X)
        
    elif model_type == 'lstm':
        # Reshape for LSTM [samples, timesteps, features]
        train_X_reshaped = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X_reshaped = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X_reshaped.shape[1], train_X_reshaped.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        # Reduced epochs for experiment speed
        model.fit(train_X_reshaped, train_y, epochs=10, batch_size=72, verbose=0, shuffle=False)
        predictions = model.predict(test_X_reshaped).flatten()

    inference_time = time.time() - start_time
    
    mae, rmse = evaluate_forecast(test_y, predictions)
    print(f"  -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, Time: {inference_time:.4f}s")
    return predictions, mae, rmse, inference_time

def main():
    dataset_path = r"c:/Users/Akshat/Desktop/AISmartHome/individual+household+electric+power+consumption/household_power_consumption.txt"
    try:
        data = load_and_process_data(dataset_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Configuration: Full Features
    features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'hour', 'day_of_week']
    
    # Configuration: 3 Days Lag (72 Hours)
    lag = 72

    # Define Experiments
    models_to_test = ['linear_regression', 'xgboost', 'lstm', 'lightgbm']
    
    results = []
    
    print(f"\nStarting Experiments with Lag = {lag} hours (3 Days)...")
    
    for model_name in models_to_test:
        try:
            _, mae, rmse, time_taken = run_experiment(data, lag, features, model_name)
            results.append({
                'Model': model_name,
                'MAE': mae,
                'RMSE': rmse,
                'Time': time_taken
            })
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            results.append({
                'Model': model_name,
                'MAE': -1,
                'RMSE': -1,
                'Time': -1
            })

    # Print Summary Table
    print("\n" + "="*60)
    print(f"{'Model':<20} | {'MAE':<10} | {'RMSE':<10} | {'Time (s)':<10}")
    print("-" * 60)
    
    for res in results:
        print(f"{res['Model']:<20} | {res['MAE']:<10.4f} | {res['RMSE']:<10.4f} | {res['Time']:<10.4f}")
    print("="*60)

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv', index=False)
    print("Results saved to results.csv")

if __name__ == "__main__":
    main()
