import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_forecast(y_true, y_pred):
    """
    Computes RMSE, MAE, and sMAPE.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # sMAPE with NaN protection (handles zero-sales days)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    smape = 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / np.where(denominator == 0, 1e-9, denominator))
    
    return {
        'rmse': rmse,
        'mae': mae,
        'smape': smape
    }
