"""
Aggregates the cleaned EV charging session data (one row per charging session)
into a daily total-energy-demand time series suitable for forecasting.

Input:  ev_charging_boulder_clean.csv (85,748 verified charging sessions, 2018-01-01 to 2023-11-30)
Output: daily_energy_demand.csv (one row per calendar day, total kWh delivered that day)

Run this first (needs only pandas, already available). Its output feeds forecast_lstm.py.
"""
import pandas as pd

SRC = "/home/claude/ev_project/ev_charging_boulder_clean.csv"
OUT = "daily_energy_demand.csv"

df = pd.read_csv(SRC, parse_dates=["Start_Date"])

# Aggregate to daily total kWh delivered across all stations.
daily = (
    df.groupby("Start_Date")
    .agg(
        total_kwh=("Energy__kWh_", "sum"),
        session_count=("Session_ID", "count"),
    )
    .reset_index()
    .sort_values("Start_Date")
)

# Fill any missing calendar days (stations with zero sessions that day) with 0 —
# a forecasting model needs a continuous daily index, not gaps.
full_range = pd.date_range(daily["Start_Date"].min(), daily["Start_Date"].max(), freq="D")
daily = daily.set_index("Start_Date").reindex(full_range, fill_value=0).rename_axis("date").reset_index()

daily.to_csv(OUT, index=False)

print(f"Rows (raw sessions): {len(df):,}")
print(f"Daily series length: {len(daily):,} days ({daily['date'].min().date()} to {daily['date'].max().date()})")
print(f"Days with zero sessions (filled): {(daily['session_count'] == 0).sum()}")
print(f"Mean daily kWh: {daily['total_kwh'].mean():.1f}  |  Std: {daily['total_kwh'].std():.1f}")
print(f"Saved -> {OUT}")
