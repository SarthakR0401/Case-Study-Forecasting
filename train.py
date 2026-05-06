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
    
    # Ensure data is sorted to prevent leakage in lags
    df = df.sort_values(['state', 'date'])
    
    states = df['state'].unique()
    best_models = {}

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
            sarima = SarimaForecaster().train(train_df['sales'])
            pred = sarima.predict(steps=FORECAST_HORIZON)
            results['sarima'] = (sarima, evaluate_forecast(y_val, pred)['rmse'])
        except Exception as e:
            print(f"SARIMA failed for {state}: {e}")

        # --- Prophet ---
        try:
            p_train = train_df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
            prophet = ProphetForecaster().train(p_train)
            pred = prophet.predict(periods=FORECAST_HORIZON)
            results['prophet'] = (prophet, evaluate_forecast(y_val, pred)['rmse'])
        except Exception as e:
            print(f"Prophet failed for {state}: {e}")

        # --- XGBoost ---
        try:
            # Drop initial rows where lags are NaN
            xgb_train = train_df.dropna(subset=X_cols)
            if not xgb_train.empty:
                xgb_model = XGBoostForecaster().train(xgb_train[X_cols], xgb_train['sales'])
                # During validation, we use val_df features (recursive is only for future)
                pred = xgb_model.predict(val_df[X_cols])
                results['xgboost'] = (xgb_model, evaluate_forecast(y_val, pred)['rmse'])
        except Exception as e:
            print(f"XGBoost failed for {state}: {e}")

        # --- LSTM ---
        try:
            lstm = LSTMForecaster().train(train_df['sales'].values)
            pred = lstm.predict(train_df['sales'].values, steps=FORECAST_HORIZON)
            results['lstm'] = (lstm, evaluate_forecast(y_val, pred)['rmse'])
        except Exception as e:
            print(f"LSTM failed for {state}: {e}")

        # Select best model for this state
        if results:
            best_model_name = min(results, key=lambda k: results[k][1])
            best_model_obj = results[best_model_name][0]
            print(f"Winner for {state}: {best_model_name} (RMSE: {results[best_model_name][1]:.2f})")
            
            # Save the best model
            joblib.dump(best_model_obj, os.path.join(MODEL_DIR, f"{state}_best_model.joblib"))
            best_models[state] = best_model_name

    # Save mapping
    joblib.dump(best_models, os.path.join(MODEL_DIR, "state_model_map.joblib"))
    print("Training pipeline finished.")

if __name__ == "__main__":
    train_and_select_best()
