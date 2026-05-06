import pandas as pd
import numpy as np

def load_and_preprocess(file_path):
    """
    Load Excel file with robust column auto-detection and handle date continuity.
    """
    df = pd.read_excel(file_path)
    
    # Standardize column names for matching
    cols = [c.strip().lower() for c in df.columns]
    
    # Robust column auto-detection
    state_candidates = ['state', 'states', 'region', 'territory']
    date_candidates = ['date', 'dates', 'day', 'period']
    sales_candidates = ['total', 'sales', 'revenue', 'amount']
    
    rename_map = {}
    for i, col in enumerate(cols):
        orig_col = df.columns[i]
        if any(cand in col for cand in state_candidates):
            rename_map[orig_col] = 'state'
        elif any(cand in col for cand in date_candidates):
            rename_map[orig_col] = 'date'
        elif any(cand in col for cand in sales_candidates):
            rename_map[orig_col] = 'sales'
            
    df = df.rename(columns=rename_map)
    
    # Final check for mandatory columns
    required = {'state', 'date', 'sales'}
    if not required.issubset(df.columns):
        # Fallback to direct substring matching if list matching failed
        for col in df.columns:
            low_col = col.lower()
            if 'state' in low_col and 'state' not in rename_map.values(): rename_map[col] = 'state'
            if 'date' in low_col and 'date' not in rename_map.values(): rename_map[col] = 'date'
            if 'sale' in low_col and 'sales' not in rename_map.values(): rename_map[col] = 'sales'
        df = df.rename(columns=rename_map)

    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns (state, date, sales). Found: {df.columns.tolist()}")

    df['date'] = pd.to_datetime(df['date'])
    return df

def handle_missing_dates(df, freq='D'):
    """Ensures continuous date sequences per state and handles gaps via interpolation."""
    df_list = []
    for state, group in df.groupby('state'):
        group = group.set_index('date').sort_index()
        all_dates = pd.date_range(start=group.index.min(), end=group.index.max(), freq=freq)
        group = group.reindex(all_dates)
        
        # Bug fix: fillna(0) replaced with interpolation + edge handling
        group['sales'] = group['sales'].interpolate(method='linear').ffill().bfill()
        group['state'] = state
        df_list.append(group.reset_index().rename(columns={'index': 'date'}))
    
    return pd.concat(df_list, ignore_index=True)
