import pandas as pd
import matplotlib.pyplot as plt
from stationarity_toolkit import StationarityToolkit

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
plot_data(ts, "Walmart Sales Data")

# Create toolkit
toolkit = StationarityToolkit(alpha=0.05)

# Test minimal verbosity
print("=== MINIMAL ===")
result = toolkit.detect(ts, verbosity='minimal')
print(result.summary)

# Test detailed verbosity with DataFrame report
print("\n=== DETAILED REPORT ===")
result = toolkit.detect(ts, verbosity='detailed')
df = result.report()
print(df)

# Save markdown to file
result.report(filepath='examples/stationarity_report.md')
print("\nMarkdown report saved to examples/stationarity_report.md")