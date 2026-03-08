# StationarityToolkit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18818474.svg)](https://doi.org/10.5281/zenodo.18818474)

A comprehensive Python library for time series stationarity analysis. Provides 10 statistical tests with clear diagnostics and actionable transformation guidance.

## What This Does

StationarityToolkit runs comprehensive stationarity checks across three dimensions: trend (unit roots, deterministic trends, structural breaks), variance (heteroskedasticity, volatility clustering), and seasonality (periodic patterns). You get 4 trend tests, 4 variance tests, and 2 seasonal tests—all in one run. Each test returns not just pass/fail, but actionable notes indicating the appropriate transformation: differencing for unit roots, detrending for deterministic trends, Box-Cox for variance instability, seasonal differencing for seasonality, or GARCH modeling for volatility clustering.

## Why Use This

Most toolkits provide limited testing and require manual interpretation of results. This toolkit runs a comprehensive suite of tests, identifies the specific type of non-stationarity (unit root vs deterministic trend vs variance instability), and suggests potential transformations. More importantly, it reveals what's actually happening in your data at each step, enabling iterative test-transform-retest workflows where you make informed decisions based on your specific use case.

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

# View summary
print(result.summary)

# View detailed results as DataFrame
result.report()

# Save report to file
result.report(filepath='stationarity_report.md')
```

## Examples

- **[Basic Usage](examples/basic_usage.py)**: Simple example showing core functionality
- **[Detailed Usage](examples/detailed_usage.ipynb)**: Comprehensive notebook demonstrating toolkit validation, transformation workflows, and real-world applications

## Statistical Tests

The toolkit runs 10 tests across three categories. Here's what each test does:

**Trend Tests (4 tests)** check for constant mean. ADF (Augmented Dickey-Fuller) tests for unit roots vs deterministic trends using 4-case logic. KPSS (Kwiatkowski-Phillips-Schmidt-Shin) complements ADF with a reverse null hypothesis. Phillips-Perron is a non-parametric unit root test robust to heteroskedasticity. Zivot-Andrews detects structural breaks (discrete regime changes, not smooth trends) and pinpoints their location.

**Variance Tests (4 tests)** check for constant variance. Levene's Test detects variance changes across segmented series. Bartlett's Test is similar to Levene but assumes normality. White's Test detects time-dependent variance patterns (heteroskedasticity). ARCH Test detects volatility clustering (time-varying conditional variance) - may trigger on trend/seasonality if present.

**Seasonal Tests (2 tests)** check for periodic patterns; contextual lag periods used based on given time-series freq. ACF/PACF Peak Detection identifies seasonality through autocorrelation peaks at contextual periods (may trigger on trend/variance). STL Decomposition tests the significance of seasonal components via variance comparison (detects deterministic seasonality).

## Workflow: How to Use the Toolkit

### Step 1: Run Initial Detection

Start by running the toolkit on your time series. Use `verbosity='detailed'` to get the full report with all test results and actionable notes.

```python
result = toolkit.detect(ts, verbosity='detailed')
```

Check the summary. If all three categories (trend, variance, seasonality) are stationary, you can proceed with modeling. If any are non-stationary, follow the transformation workflow below.

### Step 2: Address Non-Stationarity

Transformations interact unpredictably—differencing can introduce variance issues, variance stabilization can be undone by differencing, and the optimal order varies by dataset. A common starting point is Variance → Trend → Seasonality, but this isn't universal. Use the toolkit iteratively: apply a transformation, retest, and let the results guide your next step.

#### 2.1 Stabilize Variance

If Levene's, Bartlett's, or White's tests detect variance instability, consider power transformations like Box-Cox (for positive data) or Yeo-Johnson (handles negative values). If the ARCH test detects volatility clustering, you may need GARCH modeling after addressing trend and seasonality. The toolkit's notes will guide you on which issue was detected.

#### 2.2 Address Trend

If ADF, KPSS, or Phillips-Perron detect a unit root, differencing is typically needed. If they detect a deterministic trend, detrending (removing the fitted trend) is appropriate. If Zivot-Andrews detects structural breaks, consider approaches like dummy variables at break points, splitting the series into segments, or piecewise regression. The 4-case logic in the test notes will tell you which situation applies (see [4-Case Logic for Trend Tests](#4-case-logic-for-trend-tests) for details).

#### 2.3 Address Seasonality

If ACF or STL tests detect seasonal patterns, seasonal differencing at the detected period is a common approach. Alternatively, STL decomposition can separate the seasonal component, allowing you to model the residuals. The toolkit reports the detected period (e.g., 52 for yearly seasonality in weekly data).

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

### Transformation Order Considerations

Transformations interact unpredictably across datasets. Variance → Trend → Seasonality is a common starting point: unstable variance can distort trend and seasonal tests, and strong trends can obscure seasonal patterns. However, this isn't universal—sometimes seasonal differencing before regular differencing works better, or detrending before variance stabilization is more effective. The toolkit's iterative approach lets you test after each transformation and adjust based on what you observe.

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
- `detect(timeseries, verbosity='detailed')`: Run all tests and return results
  - `verbosity`: 'minimal' or 'detailed' (default: 'detailed')
  - Returns: `DetectionResult` object

### DetectionResult

**Properties**:
- `summary` (str): 3-line summary of stationarity status (Trend/Variance/Seasonal)

**Attributes**:
- `trend_stationary` (bool): True if all trend tests pass
- `variance_stationary` (bool): True if all variance tests pass
- `seasonal_stationary` (bool): True if all seasonal tests pass
- `tests` (dict): Dictionary of test results by category

**Methods**:
- `report(filepath=None)`: Generate report as DataFrame and optionally save markdown to file
  - If `filepath` provided, writes markdown report to file
  - Returns: pandas DataFrame with all test results

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

Stationarity isn't just a theoretical requirement—it's fundamental to how many time series methods work. Classical models like ARIMA and VAR are built on the assumption that your data's statistical properties don't change over time. Statistical tests like Granger causality and cointegration analysis require stationarity to produce valid results. Even machine learning models, while not strictly requiring stationarity and vary greatly by use-case, often generalize better and suffer less from distribution shift when trained on stationary data.

When you ignore non-stationarity, things go wrong in predictable ways. You get spurious regressions where variables appear related but aren't. Your forecasts become unreliable because the model learned patterns that don't hold in the future. Statistical inferences become invalid because the assumptions underlying your tests are violated. And your models become unstable, performing well on training data but failing on new data. Stationarity testing isn't about following rules—it's about ensuring your analysis reflects reality.

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
  url={https://github.com/mbsuraj/stationarityToolkit},
  doi={10.5281/zenodo.18818474}
}
```

Or use the "Cite this repository" button on GitHub for automatic citation generation.
