"""
PyTorch LSTM forecasting model for daily EV charging energy demand.

Run prepare_data.py first to generate daily_energy_demand.csv.

Requires: torch, numpy, pandas, scikit-learn, matplotlib
    pip install torch numpy pandas scikit-learn matplotlib

Usage:
    python forecast_lstm.py

Output:
    - Console: train/val/test loss curves, final test MAE/RMSE (in kWh, real units)
    - forecast_plot.png: predictions vs actuals on the held-out test set
    - loss_curve.png: training/validation loss over epochs
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("daily_energy_demand.csv", parse_dates=["date"])
series = df["total_kwh"].values.astype(np.float32).reshape(-1, 1)
dates = df["date"].values

print(f"Loaded {len(series)} daily observations "
      f"({df['date'].min().date()} to {df['date'].max().date()})")

# ---------------------------------------------------------------------------
# 2. Chronological train/val/test split (70/15/15) -- NEVER shuffle time series
# ---------------------------------------------------------------------------
n = len(series)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_raw = series[:train_end]
val_raw = series[train_end:val_end]
test_raw = series[val_end:]

print(f"Split -> train: {len(train_raw)} days | val: {len(val_raw)} days | test: {len(test_raw)} days")

# Scale using ONLY the training set's statistics to avoid leaking test info.
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_raw)

train_scaled = scaler.transform(train_raw)
val_scaled = scaler.transform(val_raw)
test_scaled = scaler.transform(test_raw)

# ---------------------------------------------------------------------------
# 3. Build sliding-window sequences: use the last LOOKBACK days to predict the next day
# ---------------------------------------------------------------------------
LOOKBACK = 14  # two weeks of history per prediction


def make_windows(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)


# Include the tail of the previous split so val/test windows have enough history.
X_train, y_train = make_windows(train_scaled, LOOKBACK)
X_val, y_val = make_windows(np.concatenate([train_scaled[-LOOKBACK:], val_scaled]), LOOKBACK)
X_test, y_test = make_windows(np.concatenate([val_scaled[-LOOKBACK:], test_scaled]), LOOKBACK)

print(f"Windowed -> train: {X_train.shape} | val: {X_val.shape} | test: {X_test.shape}")

X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_val_t = torch.from_numpy(X_val)
y_val_t = torch.from_numpy(y_val)
X_test_t = torch.from_numpy(X_test)
y_test_t = torch.from_numpy(y_test)


# ---------------------------------------------------------------------------
# 4. Model: small single-layer LSTM regressor
# ---------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]  # take the final time step's hidden state
        return self.fc(last_step)


model = LSTMForecaster()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------------------------------
# 5. Training loop with early stopping on validation loss
# ---------------------------------------------------------------------------
EPOCHS = 200
PATIENCE = 20
best_val_loss = float("inf")
patience_counter = 0
best_state = None

train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | train MSE {loss.item():.5f} | val MSE {val_loss.item():.5f}")

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
        break

# Restore best checkpoint (lowest validation loss, not just the last epoch)
model.load_state_dict(best_state)

# ---------------------------------------------------------------------------
# 6. Evaluate on held-out test set, in real kWh units (inverse-transform first)
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    test_pred_scaled = model(X_test_t).numpy()

test_pred_kwh = scaler.inverse_transform(test_pred_scaled)
test_actual_kwh = scaler.inverse_transform(y_test)

mae = mean_absolute_error(test_actual_kwh, test_pred_kwh)
rmse = np.sqrt(mean_squared_error(test_actual_kwh, test_pred_kwh))
naive_mae = mean_absolute_error(test_actual_kwh[1:], test_actual_kwh[:-1])  # baseline: predict "same as yesterday"

print("\n=== Test set results (held-out, never seen during training) ===")
print(f"MAE:  {mae:.2f} kWh")
print(f"RMSE: {rmse:.2f} kWh")
print(f"Naive baseline MAE (predict yesterday's value): {naive_mae:.2f} kWh")
print(f"Model {'beats' if mae < naive_mae else 'does NOT beat'} the naive baseline.")

# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("MSE (scaled units)")
plt.title("LSTM Training Curve — EV Charging Daily Energy Demand")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=120)
plt.close()

test_dates = pd.to_datetime(dates[val_end:val_end + len(test_pred_kwh)])
plt.figure(figsize=(12, 5))
plt.plot(test_dates, test_actual_kwh, label="Actual", linewidth=1.5)
plt.plot(test_dates, test_pred_kwh, label="Predicted", linewidth=1.5, linestyle="--")
plt.xlabel("Date")
plt.ylabel("Total daily kWh delivered")
plt.title(f"LSTM Forecast vs Actual — Test Set (MAE={mae:.1f} kWh, RMSE={rmse:.1f} kWh)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("forecast_plot.png", dpi=120)
plt.close()

print("\nSaved loss_curve.png and forecast_plot.png")
