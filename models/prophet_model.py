from prophet import Prophet
import pandas as pd

class ProphetForecaster:
    def __init__(self):
        self.model = None

    def train(self, df):
        # Prophet expects columns 'ds' and 'y'
        # Set daily_seasonality=False for daily data to avoid overfitting
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False 
        )
        self.model.fit(df)
        return self

    def predict(self, periods=56):
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        # Return only the future predictions
        return forecast['yhat'].tail(periods).values
