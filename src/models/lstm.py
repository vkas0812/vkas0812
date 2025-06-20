import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from ..preprocessing import create_sequences


def build_lstm_model(seq_len: int) -> Sequential:
    """Create a simple LSTM model."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mae")
    return model


def train_lstm(model: Sequential, series: np.ndarray, seq_len: int, epochs: int = 150, batch_size: int = 8):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1))
    X, y = create_sequences(scaled, seq_len)
    X = X.reshape(-1, seq_len, 1)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return scaler


def forecast_lstm(model: Sequential, history, steps: int, scaler: MinMaxScaler, seq_len: int):
    if hasattr(history, "values"):
        hist_vals = history.values
    else:
        hist_vals = np.asarray(history).flatten()
    hist_scaled = scaler.transform(hist_vals.reshape(-1, 1))
    seq = hist_scaled[-seq_len:].reshape(1, seq_len, 1)
    preds = []
    for _ in range(steps):
        yhat = model.predict(seq, verbose=0)[0, 0]
        preds.append(yhat)
        new_step = np.array(yhat).reshape(1, 1, 1)
        seq = np.concatenate([seq[:, 1:, :], new_step], axis=1)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
