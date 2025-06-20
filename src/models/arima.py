import pmdarima as pm
import pandas as pd


def train_arima(series: pd.Series, seasonal: bool = True, m: int = 52):
    """Fit an ARIMA model using pmdarima.auto_arima."""
    model = pm.auto_arima(
        series,
        seasonal=seasonal,
        m=m,
        suppress_warnings=True,
        stepwise=True,
    )
    return model


def forecast_arima(model, steps: int):
    """Forecast future values using a fitted ARIMA model."""
    return model.predict(n_periods=steps)
