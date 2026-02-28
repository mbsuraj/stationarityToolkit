import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
from src.stationarity_toolkit import StationarityToolkit

def plot_data(ts, title="Time Series Data"):
    """Plot the time series data."""
    plt.figure(figsize=(12, 6))
    plt.plot(ts.index, ts.values, linewidth=1.5)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Load Walmart sales data
df = pd.read_csv('examples/walmartSales_FOODS_3_586_TX_1_evaluation.csv')
df['date'] = pd.to_datetime(df['date'])
ts = pd.Series(df['value'].values, index=pd.DatetimeIndex(df['date'], freq='W-MON'))

# Plot the data
# plot_data(ts, "Walmart Sales Data")

# Create toolkit
toolkit = StationarityToolkit(alpha=0.05)

# Test minimal verbosity
print("=== MINIMAL ===")
result = toolkit.detect(ts, verbosity='minimal')
print(f"Trend stationary: {result.trend_stationary}")
print(f"Variance stationary: {result.variance_stationary}")
print(f"Seasonal stationary: {result.seasonal_stationary}")

# Test detailed verbosity with markdown report
print("\n=== DETAILED REPORT ===")
result = toolkit.detect(ts, verbosity='detailed')
print(result.report())

# Save to file
result.report(filepath='examples/stationarity_report.md')
print("\nReport saved to examples/stationarity_report.md")