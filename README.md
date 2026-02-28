# StationarityToolkit

A comprehensive Python library for time series stationarity analysis. Provides 15 statistical tests with clear diagnostics and actionable transformation guidance.

## What This Does

StationarityToolkit runs comprehensive stationarity checks across three dimensions: trend (unit roots, deterministic trends, structural breaks), variance (heteroskedasticity, volatility clustering), and seasonality (periodic patterns, seasonal unit roots). You get 6 trend tests, 4 variance tests, and 5 seasonal tests—all in one run. Each test returns not just pass/fail, but actionable notes indicating the appropriate transformation: differencing for unit roots, detrending for deterministic trends, Box-Cox for variance instability, seasonal differencing for seasonality, or GARCH modeling for volatility clustering.

## Why Use This

Most toolkits provide limited testing and require manual interpretation of results. This toolkit runs a comprehensive suite of tests, identifies the specific type of non-stationarity (unit root vs deterministic trend vs variance instability), and provides clear guidance on the appropriate transformation. If you're a data scientist, researcher, or analyst working with time series and need to prepare data for forecasting models (like ARIMA, VAR) or statistical analysis, this toolkit streamlines the diagnostic process.

## Installation

```bash
pip install stationarity-toolkit
```

**Requirements**: Python ≥3.10, pandas, numpy, scipy, statsmodels, arch

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

The toolkit runs 15 tests across three categories. Here's what each test does:

**Trend Tests (6 tests)** check for constant mean. ADF (Augmented Dickey-Fuller) tests for unit roots vs deterministic trends using 4-case logic. KPSS (Kwiatkowski-Phillips-Schmidt-Shin) complements ADF with a reverse null hypothesis. Phillips-Perron is a non-parametric unit root test robust to heteroskedasticity. Zivot-Andrews detects structural breaks and pinpoints their location (level shifts, trend shifts, or both).

**Variance Tests (4 tests)** check for constant variance. Levene's Test detects variance changes across segmented series. Bartlett's Test is similar to Levene but assumes normality. White's Test detects time-dependent variance patterns (heteroskedasticity). ARCH Test detects volatility clustering (time-varying conditional variance).

**Seasonal Tests (5 tests)** check for periodic patterns; contextual lag periods used based on given time-series freq. ACF/PACF Peak Detection identifies seasonality through autocorrelation peaks at contextual periods. STL Decomposition tests the significance of seasonal components via variance comparison. Canova-Hansen tests for seasonal unit roots (deterministic vs stochastic seasonality). OCSB is a regression-based seasonal unit root test. Spectral Analysis uses frequency-domain periodogram analysis to find periodic components.

## Workflow: How to Use the Toolkit

### Step 1: Run Initial Detection

Start by running the toolkit on your time series. Use `verbosity='detailed'` to get the full report with all test results and actionable notes.

```python
result = toolkit.detect(ts, verbosity='detailed')
```

Check the summary. If all three categories (trend, variance, seasonality) are stationary, you can proceed with modeling. If any are non-stationary, follow the transformation workflow below.

### Step 2: Address Non-Stationarity (Order Matters!)

When transforming non-stationary data, always address issues in this order: Variance → Trend → Seasonality. This sequence matters because unstable variance affects the reliability of trend and seasonal tests, and unremoved trends can obscure seasonal patterns. See [Why Order Matters](#why-order-matters-variance--trend--seasonality) for the reasoning behind this sequence.

#### 2.1 Stabilize Variance First

If Levene's, Bartlett's, or White's tests detect variance instability, consider power transformations like Box-Cox (for positive data) or Yeo-Johnson (handles negative values). If the ARCH test detects volatility clustering, you may need GARCH modeling after addressing trend and seasonality. The toolkit's notes will guide you on which issue was detected.

#### 2.2 Address Trend (After Variance is Stable)

If ADF, KPSS, or Phillips-Perron detect a unit root, differencing is typically needed. If they detect a deterministic trend, detrending (removing the fitted trend) is appropriate. If Zivot-Andrews detects structural breaks, consider approaches like dummy variables at break points, splitting the series into segments, or piecewise regression. The 4-case logic in the test notes will tell you which situation applies (see [4-Case Logic for Trend Tests](#4-case-logic-for-trend-tests) for details).

#### 2.3 Address Seasonality (Last)

If ACF, STL, Canova-Hansen, OCSB, or Spectral tests detect seasonal patterns, seasonal differencing at the detected period is a common approach. Alternatively, STL decomposition can separate the seasonal component, allowing you to model the residuals. The toolkit reports the detected period (e.g., 52 for yearly seasonality in weekly data).

After each transformation, re-run the toolkit to verify stationarity has been achieved. Sometimes multiple iterations are needed—for example, differencing might introduce new variance patterns requiring stabilization.

### Step 3: Iterate Until Stationary

After each transformation, re-run the toolkit to check if stationarity has been achieved:

```python
result = toolkit.detect(ts_transformed, verbosity='detailed')
```

Continue transforming until all three categories (trend, variance, seasonality) are stationary. Sometimes multiple iterations are needed—for example, differencing might introduce new variance patterns that require stabilization.

### Step 4: Model the Stationary Series

Once your series is stationary across all three dimensions, proceed with time series modeling. Common approaches include ARIMA/SARIMA for univariate forecasting, VAR for multivariate analysis, or linear regression with time series features. Note that some machine learning models (like tree-based methods and neural networks) can handle non-stationary data directly, but stationary data may still improve generalization and reduce distribution shift.

## Understanding Test Results

Each test returns four key pieces of information: a pass/fail result (✅ Stationary or ❌ Non-stationary), an actionable note indicating what transformation to apply if non-stationary, a brief technical interpretation with the null hypothesis (H0), and the test statistic and p-value for statistical significance.

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

## Key Concepts

### Why Order Matters: Variance → Trend → Seasonality

When transforming non-stationary data, the sequence matters. Start with variance stabilization because unstable variance throws off the reliability of trend and seasonal tests—you might miss a unit root or misidentify seasonality if variance is jumping around. Once variance is stable, address trend next. Trend affects the mean level of your series, and if you try to detect seasonality while a strong trend is present, you'll get confused signals. Remove the trend first, then seasonal patterns become clear and easy to identify. Think of it like cleaning a dataset: fix the noise (variance), remove the drift (trend), then handle the cycles (seasonality).

### Comprehensive Testing, Informed Decisions

This toolkit runs every relevant test and reports all results—not just a single pass/fail verdict. You get detailed diagnostics across trend (constant mean), variance (constant variance), and seasonality (no periodic patterns). The goal isn't to force a rigid definition of stationarity, but to give you complete information so you can make informed decisions. Maybe your use case can tolerate mild heteroskedasticity but absolutely needs no trend. Maybe you're okay with deterministic seasonality but need to eliminate stochastic trends. The toolkit shows you what's happening in your data, and you decide what matters for your specific analysis or model. Flexibility through transparency.

### 4-Case Logic for Trend Tests

When testing for trend non-stationarity, unit root tests (ADF, KPSS, Phillips-Perron) face a challenge: they need to distinguish between unit roots (random walk behavior requiring differencing) and deterministic trends (predictable drift requiring detrending). The toolkit runs each test twice—once with just a constant ('c') and once with constant plus trend ('ct'). If both pass, you're stationary. If only 'c' passes, you're stationary around a constant. If only 'ct' passes, you have a deterministic trend. If both fail, you have a unit root. 

The toolkit's report notes automatically identify which case applies and tell you the exact transformation needed:

**Example: Unit root detected (both 'c' and 'ct' fail)**
```
- Note: Unit root detected - requires differencing
- Interpretation: H0: Unit root. ADF-c p=0.2341, ADF-ct p=0.3456 >= 0.05. Fail to reject H0.
```

**Example: Deterministic trend detected ('c' fails, 'ct' passes)**
```
- Note: Deterministic trend detected - stationary after detrending
- Interpretation: H0: Unit root. ADF-c p=0.1234 >= 0.05, ADF-ct p=0.0123 < 0.05. Deterministic trend.
```

No guesswork, just clear guidance based on the 4-case analysis.

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

The toolkit automatically detects expected seasonal periods to test for seasonal non-stationarity based on your time series frequency:

- **Daily** ('D'): Tests for weekly (7), monthly (30), and yearly (365) seasonality
- **Hourly** ('H'): Tests for daily (24) and weekly (168) seasonality
- **Monthly** ('M'): Tests for quarterly (3, 6) and yearly (12) seasonality
- **Weekly** ('W'): Tests for yearly (52) seasonality
- **Quarterly** ('Q'): Tests for yearly (4) seasonality

Although the toolkit is capable of detecting time-series freq automatically, it is recommended to provide time series freq when loading the data:

```python
ts = pd.Series(values, index=pd.DatetimeIndex(dates, freq='W-MON'))
```

## Why Stationarity Matters

Many time series methods assume or benefit from stationarity:
- **Classical models**: ARIMA, VAR require stationarity
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
