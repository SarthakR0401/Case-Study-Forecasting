import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class LSTMForecaster:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.model = None
        self.scaler = MinMaxScaler()

    def _prepare_data(self, series):
        X, y = [], []
        for i in range(len(series) - self.window_size):
            X.append(series[i:i + self.window_size])
            y.append(series[i + self.window_size])
        return np.array(X), np.array(y)

    def train(self, series, epochs=30, batch_size=32):
        # Normalize data - critical for LSTM
        series = series.reshape(-1, 1)
        scaled_series = self.scaler.fit_transform(series).flatten()
        
        X, y = self._prepare_data(scaled_series)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Bug fix: activation='tanh' (standard for LSTM gates) and Dropout
        self.model = Sequential([
            LSTM(50, activation='tanh', input_shape=(self.window_size, 1), return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        return self

    def predict(self, series, steps=56):
        # Scale input
        series = series.reshape(-1, 1)
        scaled_series = self.scaler.transform(series).flatten()
        
        predictions = []
        current_seq = scaled_series[-self.window_size:].reshape((1, self.window_size, 1))
        
        for _ in range(steps):
            pred_scaled = self.model.predict(current_seq, verbose=0)[0, 0]
            predictions.append(pred_scaled)
            
            # Update sequence
            new_val = np.array([[[pred_scaled]]])
            current_seq = np.append(current_seq[:, 1:, :], new_val, axis=1)
            
        # Inverse transform
        predictions = np.array(predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(predictions).flatten()
