# Hybrid ARIMA + LSTM Forecasting

This repository contains a minimal example for building a hybrid forecasting
model.  A standard ARIMA model is first fitted to the data to capture linear
trends and seasonality.  An LSTM network is then trained on the resulting
residuals to learn any remaining non-linear structure.  The combination is
used for multi-step forecasting.

## Installation

Create a virtual environment (optional) and install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Prepare a dataset with columns `date` and `qty` in CSV or Excel format and run:

```bash
python train_hybrid.py path/to/your/data.csv
```

The script prints the MAE and RMSE of the hybrid forecast on a holdout set.
