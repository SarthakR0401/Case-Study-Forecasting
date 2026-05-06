from xgboost import XGBRegressor
import numpy as np
import pandas as pd

class XGBoostForecaster:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
        self.tail_df = None # To store historical tail for recursive prediction

    def train(self, X, y, tail_df=None):
        self.model.fit(X, y)
        self.tail_df = tail_df
        return self

    def predict(self, X=None, steps=56):
        """
        If X is provided (validation), return batch predictions.
        If X is None (future), perform recursive forecasting using self.tail_df.
        """
        if X is not None:
            return self.model.predict(X)
        
        if self.tail_df is None:
            # Fallback if no history is provided, though train() should have set it
            return np.zeros(steps)

        # Recursive Forecasting Engine
        predictions = []
        # Create a working copy of history to append new predictions
        working_df = self.tail_df.copy().sort_values('date')
        
        # X_cols mapping (must match train.py)
        X_cols = [
            'day_of_week', 'month', 'is_weekend', 'is_holiday',
            'lag_1', 'lag_7', 'lag_30', 
            'rolling_mean_7', 'rolling_std_7', 
            'rolling_mean_30', 'rolling_std_30'
        ]

        last_date = working_df['date'].max()

        for i in range(steps):
            current_date = last_date + pd.Timedelta(days=i+1)
            
            # 1. Build features for this step
            row = {
                'date': current_date,
                'day_of_week': current_date.dayofweek,
                'month': current_date.month,
                'is_weekend': 1 if current_date.dayofweek >= 5 else 0,
                # is_holiday logic simplified here; in production, use holidays library
                'is_holiday': 0 # Placeholder or pass holiday set
            }
            
            # Lags from working_df
            row['lag_1'] = working_df['sales'].iloc[-1]
            row['lag_7'] = working_df['sales'].iloc[-7] if len(working_df) >= 7 else row['lag_1']
            row['lag_30'] = working_df['sales'].iloc[-30] if len(working_df) >= 30 else row['lag_1']
            
            # Rolling from working_df
            row['rolling_mean_7'] = working_df['sales'].tail(7).mean()
            row['rolling_std_7'] = working_df['sales'].tail(7).std()
            row['rolling_mean_30'] = working_df['sales'].tail(30).mean()
            row['rolling_std_30'] = working_df['sales'].tail(30).std()
            
            # 2. Predict
            X_input = pd.DataFrame([row])[X_cols]
            pred = self.model.predict(X_input)[0]
            
            # 3. Store and update history
            predictions.append(max(0, pred)) # Clamp to positive sales
            row['sales'] = max(0, pred)
            working_df = pd.concat([working_df, pd.DataFrame([row])], ignore_index=True)
            
        return np.array(predictions)
