import pandas as pd
import glob
import os
from datetime import datetime, timedelta

# Define data paths
data_dir = 'data'
output_dir = os.path.join(data_dir, 'clean')
entso_dir = os.path.join(data_dir, 'entso-transparency-platform')
weather_file = os.path.join(data_dir, 'weather', 'weather_2015_2024.csv')

# Function to load and aggregate 15-min data to hourly
def load_and_aggregate_15min_to_hourly(file_pattern, columns_to_keep, agg_func='mean'):
    files = glob.glob(file_pattern)
    dfs = []
    for file in files:
        df = pd.read_csv(file, sep=',')
        df.columns = [col.strip('"') for col in df.columns]  # Strip quotes from column names
        df = df[columns_to_keep]
        # Convert numeric columns to float
        numeric_cols = [col for col in df.columns if col != 'MTU (CET/CEST)']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        # Parse MTU column: extract start time
        df['MTU (CET/CEST)'] = df['MTU (CET/CEST)'].str.split(' - ').str[0]
        df['MTU (CET/CEST)'] = pd.to_datetime(df['MTU (CET/CEST)'], dayfirst=True, errors='coerce')
        df.dropna(subset=['MTU (CET/CEST)'], inplace=True)
        # Resample to hourly
        df.set_index('MTU (CET/CEST)', inplace=True)
        df = df.resample('h').agg(agg_func)
        dfs.append(df)
    combined = pd.concat(dfs).sort_index()
    return combined

# Load generation data (wind/solar forecasts)
gen_columns = ['MTU (CET/CEST)', 'Day-ahead (MW)', 'Intraday (MW)', 'Current (MW)', 'Actual (MW)']
gen_df = load_and_aggregate_15min_to_hourly(
    os.path.join(entso_dir, 'generation', '*.csv'),
    gen_columns,
    agg_func='mean'
)
gen_df.columns = ['gen_day_ahead', 'gen_intraday', 'gen_current', 'gen_actual']

# Load load data
load_columns = ['MTU (CET/CEST)', 'Actual Total Load (MW)', 'Day-ahead Total Load Forecast (MW)']
load_df = load_and_aggregate_15min_to_hourly(
    os.path.join(entso_dir, 'load', '*.csv'),
    load_columns,
    agg_func='mean'
)
load_df.columns = ['load_actual', 'load_forecast']

# Load price data (already hourly, UTC)
price_files = glob.glob(os.path.join(entso_dir, 'prices', '*.csv'))
price_dfs = []
for file in price_files:
    df = pd.read_csv(file, sep=',')
    df.columns = [col.strip('"') for col in df.columns]  # Strip quotes
    df = df[['MTU (UTC)', 'Day-ahead Price (EUR/MWh)']]
    df['Day-ahead Price (EUR/MWh)'] = pd.to_numeric(df['Day-ahead Price (EUR/MWh)'], errors='coerce')
    df['MTU (UTC)'] = df['MTU (UTC)'].str.split(' - ').str[0]
    df['MTU (UTC)'] = pd.to_datetime(df['MTU (UTC)'], dayfirst=True, errors='coerce')
    df.dropna(subset=['MTU (UTC)'], inplace=True)
    price_dfs.append(df)
price_df = pd.concat(price_dfs).sort_values('MTU (UTC)')
price_df.set_index('MTU (UTC)', inplace=True)
price_df.columns = ['price']

# Load weather data (hourly)
weather_df = pd.read_csv(weather_file, parse_dates=['time'])
weather_df.set_index('time', inplace=True)
weather_df.columns = ['temp_2m', 'cloud_cover', 'wind_speed_10m', 'shortwave_radiation']

# Align timezones: ENTSO gen/load are CET/CEST, prices UTC, weather assume UTC
# Convert gen/load to UTC (subtract 1 hour for CET, but handle DST)
# For simplicity, assume all to UTC by shifting gen/load by -1 hour (ignoring DST for now)
gen_df.index = gen_df.index - timedelta(hours=1)
load_df.index = load_df.index - timedelta(hours=1)

# Merge all on index (timestamp)
merged_df = price_df.join([gen_df, load_df, weather_df], how='left')

# Handle missing values: forward fill for weather, interpolate for others
merged_df['temp_2m'] = merged_df['temp_2m'].fillna(method='ffill')
merged_df['cloud_cover'] = merged_df['cloud_cover'].fillna(method='ffill')
merged_df['wind_speed_10m'] = merged_df['wind_speed_10m'].fillna(method='ffill')
merged_df['shortwave_radiation'] = merged_df['shortwave_radiation'].fillna(method='ffill')
# For gen/load, interpolate
merged_df.interpolate(method='linear', inplace=True)

# Drop rows with missing price (target)
merged_df.dropna(subset=['price'], inplace=True)

# Add time features
merged_df['hour'] = merged_df.index.hour
merged_df['day_of_week'] = merged_df.index.dayofweek
merged_df['month'] = merged_df.index.month

# Create datasets for different periods
periods = [
    ('2022-01-01', '2025-01-01', 'dataset_2022_2025.csv'),
    ('2019-01-01', '2022-01-01', 'dataset_2019_2022.csv'),
    ('2015-01-01', '2019-01-01', 'dataset_2015_2019.csv')
]

os.makedirs(output_dir, exist_ok=True)

for start, end, filename in periods:
    subset = merged_df.loc[start:end]
    output_path = os.path.join(output_dir, filename)
    subset.to_csv(output_path)
    print(f"Created {output_path} with {len(subset)} rows")

print("Dataset creation complete.")