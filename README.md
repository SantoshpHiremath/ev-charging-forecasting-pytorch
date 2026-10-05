# EV Charging Energy Demand Forecasting (PyTorch LSTM)

## What it does

A time-series forecasting model that predicts daily total energy demand (kWh) across Boulder's public EV charging network, using an LSTM built in PyTorch. It builds on my [EV Charging Dashboard project](../ev-charging-dashboard/README.md) and reuses the same cleaned, verified dataset (85,748 real charging sessions, 2018–2023).

The project uses real data, a chronological train/validation/test split, a naive-baseline comparison, and evaluation metrics in real kWh units.

## Data

Daily total kWh delivered, aggregated from session-level records. 4 calendar days had zero sessions and were filled with 0 rather than dropped, to keep the series continuous. The model is univariate: it uses past energy values only, since Boulder's charging dataset does not include weather or grid-topology covariates.

## Methodology

- **Split**: chronological 70/15/15 train/validation/test, with no shuffling, since shuffling a time series before splitting leaks future information into training and inflates apparent accuracy.
- **Scaling**: `MinMaxScaler` fit only on the training set, then applied to validation/test, so no test-set statistics leak into training.
- **Windowing**: 14-day lookback window predicting the next day's total kWh (sliding window).
- **Model**: single-layer LSTM (32 hidden units) + linear output layer. Deliberately small, so the focus is a correctly built and correctly evaluated model.
- **Training**: Adam optimizer, MSE loss, early stopping on validation loss (patience 20 epochs, max 200 epochs), best checkpoint restored before final evaluation.
- **Evaluation**: MAE and RMSE computed in real kWh (inverse-transformed from scaled units) on the held-out test set, compared against a naive persistence baseline (predict "same as yesterday"). Reporting the baseline alongside the model's score shows whether the model adds value beyond the simplest forecast.

## Running it

```bash
pip install -r requirements.txt
```

```bash
# Step 1: aggregate the 85,748 raw charging sessions into a daily energy-demand series
python prepare_data.py

# Step 2: train and evaluate the LSTM forecaster
python forecast_lstm.py
```

Step 1 needs only pandas and produces `daily_energy_demand.csv` (2,160 daily rows, 2018-01-01 to 2023-11-30, total kWh delivered per day across all stations). Step 2 needs PyTorch and produces:

- Console output: training/validation loss per 20 epochs, early-stopping point, and final test-set MAE/RMSE in real kWh units, compared against the naive "predict yesterday's value" baseline.
- `loss_curve.png` — training vs. validation loss over epochs.
- `forecast_plot.png` — predicted vs. actual daily kWh on the held-out test period, with real dates on the x-axis.

## Results

`forecast_lstm.py` reports these metrics on the held-out test period:

- Test MAE (kWh)
- Test RMSE (kWh)
- Naive baseline MAE (kWh)
- Whether the model beats the baseline

Run the script to produce the figures for your environment.

## Notes

The lookback window (14 days), hidden size (32), and learning rate (1e-3) are reasonable defaults.

## Possible extensions

- Add covariates such as weather, topology, or operational signals for multivariate forecasting.
- Run a hyperparameter search over lookback window, hidden size, and learning rate.
- Compare against a Transformer-based model.
