import os
import pandas as pd
import numpy as np
import joblib
from utils.preprocessing import load_and_preprocess, handle_missing_dates
from utils.features import create_features
from utils.evaluate import evaluate_forecast
from models.sarima_model import SarimaForecaster
from models.prophet_model import ProphetForecaster
from models.xgboost_model import XGBoostForecaster
from models.lstm_model import LSTMForecaster
from models.exponential_smoothing_model import ExponentialSmoothingForecaster

DATA_PATH = 'data/Forecasting Case- Study.xlsx'
MODEL_DIR = 'saved_models'
FORECAST_HORIZON = 56 # 8 weeks

def train_and_select_best():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. Load and Preprocess
    print("Loading and Preprocessing data...")
    df = load_and_preprocess(DATA_PATH)
    df = handle_missing_dates(df)
    
    # 2. Feature Engineering
    df = create_features(df)
    df = df.sort_values(['state', 'date'])
    
    states = df['state'].unique()
    best_models = {}
    metrics_list = []

    # Define features to use for ML models
    X_cols = [
        'day_of_week', 'month', 'is_weekend', 'is_holiday',
        'lag_1', 'lag_7', 'lag_30', 
        'rolling_mean_7', 'rolling_std_7', 
        'rolling_mean_30', 'rolling_std_30'
    ]

    for state in states:
        print(f"Processing state: {state}")
        state_df = df[df['state'] == state].copy()
        
        # Time-series split: use last 8 weeks as validation
        train_df = state_df.iloc[:-FORECAST_HORIZON]
        val_df = state_df.iloc[-FORECAST_HORIZON:]
        y_val = val_df['sales'].values
        
        results = {}

        # --- SARIMA ---
        try:
            # Statsmodels needs a series with frequency for proper forecasting
            train_series = train_df.set_index('date')['sales']
            if train_series.index.freq is None:
                train_series.index.freq = 'D'
            
            sarima = SarimaForecaster().train(train_series)
            pred = sarima.predict(steps=FORECAST_HORIZON)
            m = evaluate_forecast(y_val, pred)
            results['sarima'] = (sarima, m['rmse'])
            metrics_list.append({'state': state, 'model': 'sarima', **m})
        except Exception as e: print(f"SARIMA error: {e}")

        # --- Prophet ---
        try:
            p_train = train_df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
            prophet = ProphetForecaster().train(p_train)
            pred = prophet.predict(periods=FORECAST_HORIZON)
            m = evaluate_forecast(y_val, pred)
            results['prophet'] = (prophet, m['rmse'])
            metrics_list.append({'state': state, 'model': 'prophet', **m})
        except Exception: 
            pass # Silent fail after our custom error message in prophet_model.py

        # --- Exponential Smoothing (ETS) ---
        try:
            # ETS also benefits from proper index
            train_series = train_df.set_index('date')['sales']
            if train_series.index.freq is None:
                train_series.index.freq = 'D'
            
            ets = ExponentialSmoothingForecaster().train(train_series)
            pred = ets.predict(steps=FORECAST_HORIZON)
            m = evaluate_forecast(y_val, pred)
            results['ets'] = (ets, m['rmse'])
            metrics_list.append({'state': state, 'model': 'ets', **m})
        except Exception as e: print(f"ETS error: {e}")

        # --- XGBoost ---
        try:
            xgb_train = train_df.dropna(subset=X_cols)
            if not xgb_train.empty:
                # Pass tail_df for recursive forecasting later
                xgb_model = XGBoostForecaster().train(xgb_train[X_cols], xgb_train['sales'], tail_df=train_df)
                pred = xgb_model.predict(val_df[X_cols])
                m = evaluate_forecast(y_val, pred)
                results['xgboost'] = (xgb_model, m['rmse'])
                metrics_list.append({'state': state, 'model': 'xgboost', **m})
        except Exception as e: print(f"XGBoost error: {e}")

        # --- LSTM ---
        try:
            lstm = LSTMForecaster().train(train_df['sales'].values)
            pred = lstm.predict(train_df['sales'].values, steps=FORECAST_HORIZON)
            m = evaluate_forecast(y_val, pred)
            results['lstm'] = (lstm, m['rmse'])
            metrics_list.append({'state': state, 'model': 'lstm', **m})
        except Exception as e: print(f"LSTM error: {e}")

        # Select best model
        if results:
            best_model_name = min(results, key=lambda k: results[k][1])
            best_model_obj = results[best_model_name][0]
            print(f"  [OK] Winner for {state}: {best_model_name}")
            
            joblib.dump(best_model_obj, os.path.join(MODEL_DIR, f"{state}_best_model.joblib"))
            best_models[state] = best_model_name

    # Save mapping and metrics
    joblib.dump(best_models, os.path.join(MODEL_DIR, "state_model_map.joblib"))
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(MODEL_DIR, "metrics_report.csv"), index=False)
    
    print("\nTraining pipeline finished.")
    print(metrics_df.groupby('model')['state'].count().rename('states_won').to_string())

if __name__ == "__main__":
    train_and_select_best()
