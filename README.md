# StationarityToolkit

A comprehensive Python library for detecting and addressing non-stationarity in time series data through rigorous statistical testing across trend, variance, and seasonality dimensions.

## Overview

StationarityToolkit provides a systematic framework for time series stationarity analysis using multiple statistical tests. The toolkit implements a strict definition of stationarity: any deviation from constant mean, constant variance, or presence of seasonality is flagged as non-stationary.

## Installation

```bash
pip install stationarity-toolkit
```

## Quick Start

```python
import pandas as pd
from stationarity_toolkit import StationarityToolkit

# Load your time series data
df = pd.read_csv('your_data.csv')
ts = pd.Series(df['value'].values, index=pd.DatetimeIndex(df['date'], freq='D'))

# Initialize toolkit
toolkit = StationarityToolkit(alpha=0.05)

# Run detection
result = toolkit.detect(ts, verbosity='detailed')

# View results
print(f"Trend stationary: {result.trend_stationary}")
print(f"Variance stationary: {result.variance_stationary}")
print(f"Seasonal stationary: {result.seasonal_stationary}")

# Generate detailed report
print(result.report())

# Save report to file
result.report(filepath='stationarity_report.md')
```

## Statistical Tests

### Trend Tests (6 tests)
- **ADF (Augmented Dickey-Fuller)**: Tests for unit roots vs deterministic trends using 4-case logic
- **KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**: Complements ADF with reverse null hypothesis
- **Phillips-Perron**: Non-parametric unit root test robust to heteroskedasticity
- **Zivot-Andrews**: Detects structural breaks (level shifts, trend shifts, or both)

### Variance Tests (4 tests)
- **Levene's Test**: Detects variance changes across segmented series
- **Bartlett's Test**: Similar to Levene but assumes normality
- **White's Test**: Detects time-dependent variance patterns (heteroskedasticity)
- **ARCH Test**: Detects volatility clustering (time-varying conditional variance)

### Seasonal Tests (5 tests)
- **ACF/PACF Peak Detection**: Identifies seasonality through autocorrelation peaks at contextual periods
- **STL Decomposition**: Tests significance of seasonal component via variance comparison
- **Canova-Hansen**: Tests for seasonal unit roots (deterministic vs stochastic seasonality)
- **OCSB**: Regression-based seasonal unit root test
- **Spectral Analysis**: Frequency-domain periodogram analysis for periodic components

## Workflow: How to Use the Toolkit

### Step 1: Run Initial Detection

```python
result = toolkit.detect(ts, verbosity='detailed')
```

Check the summary:
- If all three categories are stationary → proceed with modeling
- If any are non-stationary → follow the transformation workflow below

### Step 2: Address Non-Stationarity (Order Matters!)

**Important**: Always address issues in this order: Variance → Trend → Seasonality

#### 2.1 Stabilize Variance First

Variance instability affects the reliability of trend and seasonal tests. Address variance issues before anything else.

**Levene's Test / Bartlett's Test**
- **Detection**: Variance changes across segmented series
- **Action**: Apply Box-Cox transform (for positive data) or Yeo-Johnson transform (handles negative values)
- **Note**: Box-Cox includes log transform (λ=0), so just use Box-Cox for positive data

```python
from scipy.stats import boxcox, yeojohnson

# For positive data
ts_transformed, lambda_param = boxcox(ts)

# For data with negatives
ts_transformed, lambda_param = yeojohnson(ts)
```

**White's Test**
- **Detection**: Time-dependent variance patterns (heteroskedasticity)
- **Action**: Same as above - apply Box-Cox or Yeo-Johnson transform

**ARCH Test**
- **Detection**: Volatility clustering (GARCH effects)
- **Action**: After addressing trend/seasonality, model the variance using GARCH
- **Note**: GARCH models time-varying variance, not the mean. Use after making series stationary in mean.

```python
from arch import arch_model

# After trend/seasonality are addressed
model = arch_model(ts_stationary, vol='Garch', p=1, q=1)
result = model.fit()
```

**After variance transformation**: Re-run the toolkit to verify variance is now stationary.

#### 2.2 Address Trend (After Variance is Stable)

**ADF / KPSS / Phillips-Perron: Unit Root Detected**
- **Detection**: Series has a unit root (random walk behavior)
- **Action**: Apply first-order differencing

```python
ts_diff = ts.diff().dropna()
```

**ADF / KPSS / Phillips-Perron: Deterministic Trend Detected**
- **Detection**: Series has a deterministic (predictable) trend
- **Action**: Detrend by subtracting fitted trend

```python
from scipy.signal import detrend

ts_detrended = pd.Series(detrend(ts.values), index=ts.index)
```

**Zivot-Andrews: Structural Break Detected**
- **Detection**: Structural break at specific observation (level shift, trend shift, or both)
- **Action**: Handle breaks using one of these approaches:
  1. Add dummy variables at break points
  2. Split series into segments and model separately
  3. Use piecewise regression

```python
# Example: dummy variable approach
break_point = 99  # from ZA test output
ts['break_dummy'] = (ts.index >= break_point).astype(int)
```

**After trend transformation**: Re-run the toolkit to verify trend stationarity.

#### 2.3 Address Seasonality (Last)

**ACF / STL / Canova-Hansen / OCSB / Spectral: Seasonality Detected**
- **Detection**: Seasonal patterns at detected period (e.g., 52 for weekly data with yearly seasonality)
- **Action**: Apply seasonal differencing

```python
# For period = 52 (yearly seasonality in weekly data)
ts_seasonal_diff = ts.diff(periods=52).dropna()
```

**Alternative**: Use STL decomposition and model the residuals

```python
from statsmodels.tsa.seasonal import STL

stl = STL(ts, period=52, seasonal=13)
result = stl.fit()
ts_residual = result.resid  # Model this
```

**After seasonal transformation**: Re-run the toolkit to verify seasonal stationarity.

### Step 3: Iterate Until Stationary

After each transformation, re-run the toolkit:

```python
result = toolkit.detect(ts_transformed, verbosity='detailed')
```

Continue transforming until all three categories (trend, variance, seasonality) are stationary.

### Step 4: Model the Stationary Series

Once stationary, proceed with your time series modeling:
- ARIMA / SARIMA
- VAR (Vector Autoregression)
- Linear regression with time series features
- Machine learning models (LSTM, XGBoost, etc.)

## Understanding Test Results

Each test returns:
- **Result**: ✅ Stationary or ❌ Non-stationary
- **Note**: Actionable guidance on what to do if non-stationary
- **Interpretation**: Brief technical interpretation with null hypothesis (H0)
- **Statistic**: Test statistic value
- **P-value**: Statistical significance

### Example Output

```
### Augmented Dickey-Fuller (ADF) Test
- Result: ✅ Stationary
- Note: Stationary around constant mean
- Interpretation: H0: Unit root present. ADF-c p=0.0234, ADF-ct p=0.0456 < 0.05. Reject H0.
- Statistic: -3.2145
- P-value: 0.0234
```

## Key Concepts

### Why Order Matters: Variance → Trend → Seasonality

1. **Variance first**: Unstable variance affects the reliability of all other tests. Stabilizing variance ensures accurate trend and seasonal detection.

2. **Trend second**: Trend affects the mean level. Removing trend before seasonality ensures seasonal patterns are properly identified.

3. **Seasonality last**: Seasonal patterns are easier to detect and remove once variance is stable and trend is removed.

### Strict Stationarity Definition

This toolkit uses a strict definition:
- **Trend stationary**: Constant mean (no unit roots, no deterministic trends, no structural breaks)
- **Variance stationary**: Constant variance (no heteroskedasticity, no volatility clustering)
- **Seasonal stationary**: No seasonal patterns (no periodic components at any frequency)

### 4-Case Logic for Unit Root Tests

ADF, KPSS, and Phillips-Perron tests run both 'c' (constant) and 'ct' (constant + trend) specifications:

1. **Both stationary** → Series is stationary
2. **'c' stationary, 'ct' not** → Series is stationary (around constant)
3. **'c' not, 'ct' stationary** → Deterministic trend present (detrend needed)
4. **Both non-stationary** → Unit root present (differencing needed)

## API Reference

### StationarityToolkit

```python
toolkit = StationarityToolkit(alpha=0.05)
```

**Parameters**:
- `alpha` (float): Significance level for all tests (default: 0.05)

**Methods**:
- `detect(timeseries, verbosity='minimal')`: Run all tests and return results
  - `verbosity`: 'minimal' or 'detailed'
  - Returns: `DetectionResult` object

### DetectionResult

**Attributes**:
- `trend_stationary` (bool): True if all trend tests pass
- `variance_stationary` (bool): True if all variance tests pass
- `seasonal_stationary` (bool): True if all seasonal tests pass
- `trend_results` (list): Individual trend test results
- `variance_results` (list): Individual variance test results
- `seasonal_results` (list): Individual seasonal test results

**Methods**:
- `report(filepath=None)`: Generate markdown report
  - If `filepath` provided, saves to file
  - Returns: Report string

## Contextual Period Detection

The toolkit automatically detects expected seasonal periods based on your time series frequency:

- **Daily** ('D'): Tests for weekly (7), monthly (30), and yearly (365) seasonality
- **Hourly** ('H'): Tests for daily (24) and weekly (168) seasonality
- **Monthly** ('M'): Tests for quarterly (3, 6) and yearly (12) seasonality
- **Weekly** ('W'): Tests for yearly (52) seasonality
- **Quarterly** ('Q'): Tests for yearly (4) seasonality

Ensure your time series has a proper frequency set:

```python
ts = pd.Series(values, index=pd.DatetimeIndex(dates, freq='W-MON'))
```

## Why Stationarity Matters

Many time series methods assume or benefit from stationarity:
- **Classical models**: ARIMA, VAR, exponential smoothing require stationarity
- **Statistical tests**: Granger causality, cointegration assume stationarity
- **Machine learning**: While not strictly required, stationary data can improve generalization and reduce distribution shift

Non-stationary data can lead to:
- Spurious regressions (false relationships)
- Poor forecast accuracy
- Invalid statistical inferences
- Model instability

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{stationarity_toolkit,
  title={StationarityToolkit: Comprehensive Time Series Stationarity Analysis},
  author={Malla, Bhanu Suraj},
  year={2024},
  url={https://github.com/mbsuraj/stationarityToolkit}
}
```

Or use the "Cite this repository" button on GitHub for automatic citation generation.
