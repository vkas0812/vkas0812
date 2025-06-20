import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .arima import train_arima, forecast_arima
from .lstm import build_lstm_model, forecast_lstm
from ..preprocessing import create_sequences


def train_hybrid(train_series: pd.Series, seq_len: int = 5, epochs: int = 150, seasonal: bool = True, m: int = 52):
    """Train ARIMA on the series and LSTM on its residuals."""
    arima_model = train_arima(train_series, seasonal=seasonal, m=m)
    residuals = train_series - arima_model.predict_in_sample()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))
    X, y = create_sequences(scaled, seq_len)
    X = X.reshape(-1, seq_len, 1)

    lstm_model = build_lstm_model(seq_len)
    lstm_model.fit(X, y, epochs=epochs, batch_size=8, verbose=0)

    return arima_model, lstm_model, scaler


def forecast_hybrid(arima_model, lstm_model, scaler, history: pd.Series, steps: int, seq_len: int = 5):
    base_pred = forecast_arima(arima_model, steps)
    res_pred = forecast_lstm(lstm_model, history, steps, scaler, seq_len)
    return base_pred + res_pred
