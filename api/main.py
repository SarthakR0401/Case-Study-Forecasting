from fastapi import FastAPI, HTTPException
from api.schemas import ForecastRequest, ForecastResponse
from api.model_loader import load_model_for_state
from utils.preprocessing import load_and_preprocess, handle_missing_dates
from utils.features import create_features
import pandas as pd
from datetime import timedelta
import os
import numpy as np
from contextlib import asynccontextmanager

# Global cache
DATA_CACHE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data into memory once at startup."""
    print("Pre-loading data for API...")
    path = 'data/Forecasting Case- Study.xlsx'
    if os.path.exists(path):
        df = load_and_preprocess(path)
        df = handle_missing_dates(df)
        df = create_features(df)
        DATA_CACHE['full_df'] = df
        
        # Pre-load metrics if available
        if os.path.exists('saved_models/metrics_report.csv'):
            DATA_CACHE['metrics'] = pd.read_csv('saved_models/metrics_report.csv').to_dict('records')
    else:
        print("Warning: Data file not found at startup.")
    yield
    DATA_CACHE.clear()

app = FastAPI(title="Production Forecasting System API (v2.0)", lifespan=lifespan)

@app.get("/")
def health_check():
    return {"status": "healthy", "version": "2.0"}

@app.get("/states")
def list_states():
    """Returns all states available in the dataset."""
    if 'full_df' not in DATA_CACHE:
        raise HTTPException(status_code=500, detail="Data not loaded.")
    return sorted(DATA_CACHE['full_df']['state'].unique().tolist())

@app.get("/metrics")
def get_metrics():
    """Returns training performance report."""
    return DATA_CACHE.get('metrics', {"error": "Metrics report not found."})

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
    
    lower, upper = None, None
    
    try:
        if model_name == 'prophet':
            preds, lower, upper = model.predict(periods=56, return_conf_int=True)
        elif model_name == 'sarima':
            preds, lower, upper = model.predict(steps=56, return_conf_int=True)
        elif model_name == 'xgboost':
            preds = model.predict(steps=56)
            # XGBoost doesn't natively give CI, we'll simulate a 5% band for visualization
            lower = preds * 0.95
            upper = preds * 1.05
        elif model_name == 'lstm':
            preds = model.predict(state_df['sales'].values, steps=56)
            lower = preds * 0.90
            upper = preds * 1.10
        else:
            preds = np.zeros(56)
            
        return {
            "state": state,
            "model_used": model_name,
            "forecast": [float(p) for p in preds],
            "dates": forecast_dates,
            "lower": [float(l) for l in lower] if lower is not None else None,
            "upper": [float(u) for u in upper] if upper is not None else None,
            "history": state_df['sales'].tail(30).tolist(),
            "history_dates": [d.strftime('%Y-%m-%d') for d in state_df['date'].tail(30)]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
