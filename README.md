# EV Charging Energy Demand Forecasting (PyTorch LSTM)

A time-series forecasting model predicting daily total energy demand (kWh) across Boulder's public
EV charging network, using an LSTM built in PyTorch. Built to gain real, hands-on deep-learning and
time-series experience on top of the existing [EV Charging Dashboard project](../ev_project/README.md),
reusing the same cleaned, verified dataset (85,748 real charging sessions, 2018–2023).

## Why this exists

Built specifically to close a real gap: hands-on PyTorch and time-series forecasting experience,
as opposed to just listing PyTorch as a CV skill. This uses genuine data, a proper chronological
train/val/test split, an honest baseline comparison, and real evaluation metrics — not a toy example.

## Setup

```bash
pip install -r requirements.txt
```

## How to run

```bash
# Step 1: aggregate the 85,748 raw charging sessions into a daily energy-demand series
python prepare_data.py

# Step 2: train and evaluate the LSTM forecaster
python forecast_lstm.py
```

Step 1 needs only pandas and produces `daily_energy_demand.csv` (2,160 daily rows, 2018-01-01 to
2023-11-30, total kWh delivered per day across all stations). Step 2 needs PyTorch and produces:

- Console output: training/validation loss per 20 epochs, early-stopping point, and final test-set
  MAE/RMSE in real kWh units, compared against a naive "predict yesterday's value" baseline.
- `loss_curve.png` — training vs. validation loss over epochs.
- `forecast_plot.png` — predicted vs. actual daily kWh on the held-out test period, with real dates
  on the x-axis.

## Methodology

- **Data**: daily total kWh delivered, aggregated from session-level records. 4 calendar days had
  zero sessions and were filled with 0 rather than dropped, to keep the series continuous (a real
  data-quality note, not swept under the rug).
- **Split**: chronological 70/15/15 train/validation/test — no shuffling, since shuffling a time
  series before splitting leaks future information into training and silently inflates apparent
  accuracy.
- **Scaling**: `MinMaxScaler` fit only on the training set, then applied to validation/test — fitting
  the scaler on the full dataset would leak test-set statistics into training.
- **Windowing**: 14-day lookback window predicting the next day's total kWh (sliding window).
- **Model**: single-layer LSTM (32 hidden units) + linear output layer. Deliberately small — the
  goal is a correctly-built, correctly-evaluated model, not maximum capacity.
- **Training**: Adam optimizer, MSE loss, early stopping on validation loss (patience 20 epochs,
  max 200 epochs), best checkpoint restored before final evaluation.
- **Evaluation**: MAE and RMSE computed in real kWh (inverse-transformed from scaled units) on the
  held-out test set, compared against a naive persistence baseline (predict "same as yesterday").
  A forecasting model that can't beat this simple baseline isn't adding real value — reporting the
  baseline alongside the model's score is standard practice and avoids overselling the result.

## Honest limitations

- Univariate: only past energy values are used. The job posting this was built for (Siemens —
  time-series forecasting for energy systems) asks about *multivariate* forecasting with covariates
  like weather, topology, or operational signals — this project deliberately does not attempt that,
  since Boulder's charging dataset doesn't include weather or grid-topology data. It should be
  described honestly as a univariate baseline, not multivariate forecasting experience.
- No hyperparameter search — lookback window (14 days), hidden size (32), and learning rate (1e-3)
  are reasonable defaults, not tuned.
- No Transformer-based model — this is deliberately scoped as a solid single-model LSTM build rather
  than a broader architecture comparison.
- Results (MAE/RMSE numbers) are filled in after running the script locally — see below.

## Results

*(Fill in after running `forecast_lstm.py` locally — paste the console output here.)*

- Test MAE: **TBD kWh**
- Test RMSE: **TBD kWh**
- Naive baseline MAE: **TBD kWh**
- Model beats baseline: **TBD**
