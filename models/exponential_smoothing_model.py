from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import numpy as np

class ExponentialSmoothingForecaster:
    def __init__(self, seasonal_periods=7):
        self.seasonal_periods = seasonal_periods
        self.model_fit = None

    def train(self, series):
        # Using additive trend and seasonality as a robust default
        # Use a Series with a proper index/frequency for best results
        self.model = ExponentialSmoothing(
            series, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated"
        )
        self.model_fit = self.model.fit()
        return self

    def predict(self, steps=56):
        if self.model_fit:
            preds = self.model_fit.forecast(steps)
            return preds.values
        return np.zeros(steps)
