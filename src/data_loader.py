import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Read a CSV or Excel file with a 'date' column."""
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path, parse_dates=['date'])
    else:
        df = pd.read_excel(path, parse_dates=['date'])
    return df
