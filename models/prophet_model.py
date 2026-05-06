from prophet import Prophet
import pandas as pd
import numpy as np

class ProphetForecaster:
    def __init__(self):
        self.model = None

    def train(self, df):
        # Prophet expects columns 'ds' and 'y'
        # Ensure data types are strictly correct to avoid backend issues
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        try:
            # daily_seasonality=False to avoid overfitting on daily data
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            self.model.fit(df)
        except Exception as e:
            error_msg = str(e)
            if 'stan_backend' in error_msg or 'CmdStan' in error_msg:
                print("\n[!] Prophet Error: CmdStan installation not found or backend failure.")
                print("    To fix this, run: python -m cmdstanpy.install_cmdstan")
            raise e
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
