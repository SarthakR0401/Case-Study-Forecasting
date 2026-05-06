from prophet import Prophet
import pandas as pd
import numpy as np

class ProphetForecaster:
    def __init__(self):
        self.model = None

    def train(self, df):
        # Prophet expects columns 'ds' and 'y'
        # daily_seasonality=False to avoid overfitting on daily data
        # seasonality_mode='multiplicative' for sales patterns
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        self.model.fit(df)
        return self

    def predict(self, periods=56, return_conf_int=False):
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        # Return only the future predictions
        preds = forecast['yhat'].tail(periods).values
        if return_conf_int:
            lower = forecast['yhat_lower'].tail(periods).values
            upper = forecast['yhat_upper'].tail(periods).values
            return preds, lower, upper
        return preds
