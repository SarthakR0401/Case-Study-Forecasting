from fastapi import FastAPI, HTTPException
from api.schemas import ForecastRequest, ForecastResponse
from api.model_loader import load_model_for_state
from utils.preprocessing import load_and_preprocess, handle_missing_dates
from utils.features import create_features
import pandas as pd
from datetime import timedelta
import os
import numpy as np

app = FastAPI(title="Production Forecasting System API")

# Global state to avoid slow I/O on every request
DATA_CACHE = {}

@app.on_event("startup")
def startup_event():
    """Load data into memory once at startup."""
    print("Pre-loading data for API...")
    if os.path.exists('data/Forecasting Case- Study.xlsx'):
        df = load_and_preprocess('data/Forecasting Case- Study.xlsx')
        df = handle_missing_dates(df)
        df = create_features(df)
        DATA_CACHE['full_df'] = df
    else:
        print("Warning: Data file not found at startup.")

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "forecasting-system"}

@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    state = request.state
    
    if 'full_df' not in DATA_CACHE:
        raise HTTPException(status_code=500, detail="Data not loaded on server.")
    
    model, model_name = load_model_for_state(state)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found for state: {state}.")
    
    state_df = DATA_CACHE['full_df'][DATA_CACHE['full_df']['state'] == state].copy()
    last_date = state_df['date'].max()
    forecast_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 57)]
    
    try:
        if model_name == 'prophet':
            preds = model.predict(periods=56)
        elif model_name == 'sarima':
            preds = model.predict(steps=56)
        elif model_name == 'xgboost':
            # XGBoost Recursive Forecast
            X_cols = [
                'day_of_week', 'month', 'is_weekend', 'is_holiday',
                'lag_1', 'lag_7', 'lag_30', 
                'rolling_mean_7', 'rolling_std_7', 
                'rolling_mean_30', 'rolling_std_30'
            ]
            
            # Simple recursive: we'll use the last known features as a baseline
            # In a full prod version, we would recompute lags step-by-step
            last_features = state_df[X_cols].tail(1)
            preds = []
            for _ in range(56):
                p = model.predict(last_features)[0]
                preds.append(p)
                # Update features slightly for the next step (very simplified)
                last_features['lag_1'] = p 
        elif model_name == 'lstm':
            preds = model.predict(state_df['sales'].values, steps=56)
        else:
            preds = [0.0] * 56
            
        return {
            "state": state,
            "model_used": model_name,
            "forecast": [float(p) for p in preds],
            "dates": forecast_dates,
            "history": state_df['sales'].tail(30).tolist(),
            "history_dates": [d.strftime('%Y-%m-%d') for d in state_df['date'].tail(30)]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
