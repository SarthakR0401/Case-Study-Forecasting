import pandas as pd
import numpy as np
import holidays

def create_features(df):
    """
    Feature engineering for time series forecasting.
    Includes lags, rolling windows, and holiday flags.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['state', 'date'])

    # US Holiday Flags
    us_holidays = holidays.US()
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in us_holidays else 0)

    # Calendar features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Per-state features
    processed_dfs = []
    for state, group in df.groupby('state'):
        group = group.copy().sort_values('date')
        
        # Lags
        group['lag_1'] = group['sales'].shift(1)
        group['lag_7'] = group['sales'].shift(7)
        group['lag_30'] = group['sales'].shift(30)
        
        # Rolling features
        group['rolling_mean_7'] = group['sales'].shift(1).rolling(window=7).mean()
        group['rolling_std_7'] = group['sales'].shift(1).rolling(window=7).std()
        group['rolling_mean_30'] = group['sales'].shift(1).rolling(window=30).mean()
        group['rolling_std_30'] = group['sales'].shift(1).rolling(window=30).std()
        
        processed_dfs.append(group)

    return pd.concat(processed_dfs, ignore_index=True)
