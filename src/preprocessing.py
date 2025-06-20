import numpy as np
import pandas as pd


def remove_outliers_interpolate(df: pd.DataFrame, column: str = "qty") -> pd.DataFrame:
    """Remove outliers using the IQR method and interpolate missing values."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    cleaned = df.copy()
    mask = (cleaned[column] < lower_bound) | (cleaned[column] > upper_bound)
    cleaned.loc[mask, column] = np.nan

    cleaned = cleaned.set_index("date")
    cleaned[column].interpolate(method="time", inplace=True)
    return cleaned.reset_index()


def resample_weekly(df: pd.DataFrame, column: str = "qty") -> pd.DataFrame:
    """Resample data by week, summing the quantity column."""
    return (
        df.set_index("date")[column]
        .resample("W")
        .sum()
        .fillna(0)
        .reset_index()
    )


def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i : i + seq_len])
        ys.append(data[i + seq_len])
    return np.array(xs), np.array(ys)
