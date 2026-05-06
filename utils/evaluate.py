import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_forecast(y_true, y_pred):
    """
    Computes RMSE, MAE, MAPE, and sMAPE with safety guards.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (skip zeros to avoid Inf)
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
        
    # sMAPE with NaN protection (0+0 denominator replaced with 1e-8)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    smape = 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / np.where(denominator == 0, 1e-8, denominator))
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'smape': float(smape)
    }
