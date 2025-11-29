import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_process_data(filepath):
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load data
    df = pd.read_csv(filepath, sep=';', 
                     parse_dates={'dt': ['Date', 'Time']}, 
                     infer_datetime_format=True, 
                     dayfirst=True,
                     low_memory=False, 
                     na_values=['nan', '?'])

    df = df.set_index('dt')
    df = df.fillna(method='ffill')

    # Convert columns to numeric
    cols_to_numeric = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Feature Engineering for Analysis
    df['Hour'] = df.index.hour
    df['Month'] = df.index.month
    df['Weekday'] = df.index.dayofweek
    
    # Resample to Hourly for clearer visualization (and consistency with our models)
    df_hourly = df.resample('H').mean()
    
    return df_hourly

def perform_eda(df):
    print("Performing EDA...")
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap of Energy Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    print("Saved correlation_heatmap.png")
    
    # 2. Daily Profile (Average Day)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Hour', y='Global_active_power', data=df)
    plt.title('Average Daily Energy Consumption Profile')
    plt.xlabel('Hour of Day')
    plt.ylabel('Global Active Power (kW)')
    plt.grid(True)
    plt.savefig('daily_profile.png')
    print("Saved daily_profile.png")

    # 3. Weekly Profile
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Weekday', y='Global_active_power', data=df)
    plt.title('Energy Consumption by Day of Week (0=Mon, 6=Sun)')
    plt.xlabel('Day of Week')
    plt.ylabel('Global Active Power (kW)')
    plt.grid(True)
    plt.savefig('weekly_profile.png')
    print("Saved weekly_profile.png")
    
    # 4. Statistical Summary
    summary = df.describe()
    summary.to_csv('data_summary.csv')
    print("Saved data_summary.csv")
    
    return correlation_matrix

def main():
    dataset_path = r"c:/Users/Akshat/Desktop/AISmartHome/individual+household+electric+power+consumption/household_power_consumption.txt"
    try:
        df = load_and_process_data(dataset_path)
        corr_matrix = perform_eda(df)
        
        # Print insights for the report
        print("\n--- Correlation with Target (Global_active_power) ---")
        print(corr_matrix['Global_active_power'].sort_values(ascending=False))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
