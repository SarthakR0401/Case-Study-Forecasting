from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

class SarimaForecaster:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None

    def train(self, series):
        model = SARIMAX(series, order=self.order, seasonal_order=self.seasonal_order)
        self.model_fit = model.fit(disp=False)
        return self

    def predict(self, steps=56):
        if self.model_fit:
            return self.model_fit.forecast(steps=steps)
        return None
