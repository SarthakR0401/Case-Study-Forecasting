import pandas as pd
import numpy as np

def load_and_preprocess(file_path):
    """
    Load Excel file, clean data, and handle missing values/dates.
    """
    df = pd.read_excel(file_path)
    
    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Robust column mapping to handle variations in Excel headers
    rename_map = {
        'state': 'state',
        'date': 'date',
        'total': 'sales', 
        'sales': 'sales'
    }
    df = df.rename(columns=rename_map)
    
    # Ensure mandatory columns exist
    required = {'state', 'date', 'sales'}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")

    df['date'] = pd.to_datetime(df['date'])
    return df

def handle_missing_dates(df, freq='D'):
    """Ensures continuous date sequences per state and handles missing values."""
    df_list = []
    for state, group in df.groupby('state'):
        group = group.set_index('date').sort_index()
        all_dates = pd.date_range(start=group.index.min(), end=group.index.max(), freq=freq)
        group = group.reindex(all_dates)
        
        # Smarter missing value handling: Interpolation + Forward Fill + Fallback to 0
        group['sales'] = group['sales'].interpolate(method='linear').ffill().fillna(0)
        group['state'] = state
        df_list.append(group.reset_index().rename(columns={'index': 'date'}))
    
    return pd.concat(df_list, ignore_index=True)
