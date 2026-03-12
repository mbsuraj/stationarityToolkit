# StationarityToolkit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18818474.svg)](https://doi.org/10.5281/zenodo.18818474)

Python library for time series stationarity analysis. Runs 10 statistical tests across trend, variance, and seasonality dimensions.

## Installation

```bash
pip install stationaritytoolkit
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

## Statistical Tests

**Trend Tests (4 tests)** check for constant mean:
- ADF (Augmented Dickey-Fuller): Tests for unit roots. Runs both constant-only ('c') and constant+trend ('ct') specifications to distinguish unit roots from deterministic trends.
- KPSS (Kwiatkowski-Phillips-Schmidt-Shin): Complements ADF with reverse null hypothesis. Runs 'c' and 'ct' modes.
- Phillips-Perron: Non-parametric unit root test robust to heteroskedasticity. Runs 'c' and 'ct' specifications.
- Zivot-Andrews: Detects structural breaks and pinpoints their location.

**Variance Tests (4 tests)** check for constant variance:
- Levene's Test: Detects variance changes across time segments (robust to non-normality).
- Bartlett's Test: Similar to Levene but assumes normality.
- White's Test: Detects time-dependent variance patterns.
- ARCH Test: Detects volatility clustering.

**Seasonal Tests (2 tests)** check for periodic patterns using contextual periods based on time series frequency:
- ACF/PACF Peak Detection: Uses Ljung-Box test on seasonal lags to detect autocorrelation at periodic intervals (may trigger on trend/variance).
- STL Decomposition: Tests significance of seasonal components via variance comparison (detects deterministic seasonality).

## Usage

Run the toolkit on your time series:

```python
result = toolkit.detect(ts, verbosity='detailed')
```

## Understanding Test Results

Each test returns: pass/fail result (✅ Stationary or ❌ Non-stationary), actionable note, technical interpretation with null hypothesis (H0), and test statistic with p-value.

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

### Transformation Order Considerations

Transformations interact unpredictably across datasets. Variance → Trend → Seasonality is a common starting point: unstable variance can distort trend and seasonal tests, and strong trends can obscure seasonal patterns. However, this isn't universal—sometimes seasonal differencing before regular differencing works better, or detrending before variance stabilization is more effective. The toolkit's iterative approach lets you test after each transformation and adjust based on what you observe.

### Comprehensive Testing, Informed Decisions

This toolkit runs every relevant test and reports all results—not just a single pass/fail verdict. You get detailed diagnostics across trend (constant mean), variance (constant variance), and seasonality (no periodic patterns). The goal isn't to force a rigid definition of stationarity, but to give you complete information so you can make informed decisions. Maybe your use case can tolerate mild heteroskedasticity but absolutely needs no trend. Maybe you're okay with deterministic seasonality but need to eliminate stochastic trends. The toolkit shows you what's happening in your data, and you decide what matters for your specific analysis or model.

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

## License

MIT License

## Contributing

Contributions welcome! We're particularly interested in expanding the test suite with additional stationarity tests. If you know of relevant statistical tests for trend, variance, or seasonality detection, please open an issue to discuss or submit a pull request.

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
