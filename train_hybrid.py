import sys
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_loader import load_dataset
from src.preprocessing import remove_outliers_interpolate, resample_weekly
from src.models.hybrid import train_hybrid, forecast_hybrid


def main(path: str):
    df = load_dataset(path)
    df = remove_outliers_interpolate(df)
    df = resample_weekly(df)

    test_horizon = 31
    train_df = df[:-test_horizon]
    test_df = df[-test_horizon:]

    arima_model, lstm_model, scaler = train_hybrid(train_df["qty"], seq_len=5, epochs=10)
    preds = forecast_hybrid(arima_model, lstm_model, scaler, train_df["qty"], len(test_df), seq_len=5)

    mae = mean_absolute_error(test_df["qty"], preds)
    rmse = np.sqrt(mean_squared_error(test_df["qty"], preds))
    print(f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_hybrid.py <path_to_dataset>")
        sys.exit(1)
    main(sys.argv[1])
