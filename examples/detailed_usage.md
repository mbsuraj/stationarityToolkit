# Stationarity Toolkit: Comprehensive Time Series Analysis

Stationarity testing is typically done with one or two chosen tests . But here's the problem: different tests detect different types of non-stationarity. A series might pass ADF / KPSS but fail Zivot-Andrews because one tests for unit rootes, detereministic trends while the other tests for structural breaks. Your variance might look stable when split into segments (Levene test passes) but show clear time-dependent patterns (White test fails). Likewise for seasonality, it could be a fixed repeating pattern or a random-walk seasonal component - you need different tests to tell them apart.

The **Stationarity Toolkit** addresses this by running 10 statistical tests across three categories: trend, variance, and seasonality. Instead of getting a single yes/no answer, you get a complete picture of what's happening in your time series.

But comprehensive testing reveals another challenge: transformations interact in unpredictable ways. You might difference your data to remove trend, only to discover you've introduced variance issues that weren't there before. Or you stabilize variance with Box-Cox, then find that differencing undoes your work. This is where the toolkit becomes essential - it lets you test after each transformation to see what actually happened, not what you assumed would happen.

This notebook walks through the toolkit's capabilities in four parts. We start with synthetic data where we know exactly what non-stationarity exists, then move to increasingly complex scenarios, and finally tackle real-world data where nothing behaves quite like the textbooks say it should.

For complete details on the 10 tests and their interpretations, see the [README](README.md).


```python
# Install toolkit
# !pip install stationaritytoolkit
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
from stationarity_toolkit.toolkit import StationarityToolkit

# Set working directory to repo root
if os.path.basename(os.getcwd()) == 'examples':
    os.chdir('..')

# Configure pandas display for full text visibility
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# Suppress interpolation warnings from statsmodels
warnings.filterwarnings('ignore', category=Warning)

np.random.seed(42)
toolkit = StationarityToolkit(alpha=0.05)

def plot_ts(values, title):
    plt.figure(figsize=(12, 4))
    plt.plot(values)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()
```

## Part 1: Does the Toolkit Actually Work?

Before we trust the toolkit with real data, let's validate it on synthetic data where we know exactly what's going on. We'll create six time series with specific properties and see if the tests catch what they're supposed to catch.

We start simple - pure noise and noise with a constant baseline. Both should be stationary, and all tests should pass. Then we add a trend, which should trigger the trend tests. Next, seasonality, which should trigger the seasonal tests. Then we test two types of variance non-stationarity: smooth growth (which most variance tests should catch) and clustered volatility (which specifically the ARCH test should catch).

But validation isn't just about confirming tests work - it's also about discovering their limitations. You'll see the Zivot-Andrews test find "structural breaks" in smooth trends where none exist. You'll see the ARCH test fail when there's trend or seasonality, even though there's no volatility clustering. You'll see ACF confuse variance issues with seasonality. These aren't bugs - they're inherent limitations that you need to understand when interpreting results on real data.

### 1.1 Pure Noise (Stationary Baseline)


```python
n = 1000
ts_noise = pd.Series(
    np.random.normal(0, 1, n),
    index=pd.date_range('2020-01-01', periods=n, freq='D')
)
plot_ts(ts_noise.values, "Pure Noise (Stationary)")

result = toolkit.detect(ts_noise, verbosity='detailed')
print(result.summary)
result.report(filepath='examples/case1_noise_report.md')
```


    
![png](detailed_usage_files/detailed_usage_5_0.png)
    


    Trend Stationary: ✅ Yes
    Variance Stationary: ✅ Yes
    Seasonal Stationary: ✅ Yes





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>✅ Stationary</td>
      <td>-31.8111</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. ADF-c p=0.0000 &lt; 0.05, ADF-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>✅ Stationary</td>
      <td>0.1871</td>
      <td>0.1000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Stationary. KPSS-c p=0.1000 &gt; 0.05, KPSS-ct p=0.1000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>✅ Stationary</td>
      <td>-31.8105</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. PP-c p=0.0000 &lt; 0.05, PP-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-32.0778</td>
      <td>0.0000</td>
      <td>Stationary with structural breaks: level shift at obs 428, trend shift at obs 157, level+trend shift at obs 428. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0000, ZA-t p=0.0007, ZA-ct p=0.0002 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>0.2055</td>
      <td>0.8926</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments. Levene p=0.8926 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>0.5547</td>
      <td>0.9067</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.9067 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>✅ Stationary</td>
      <td>0.8660</td>
      <td>0.6486</td>
      <td>Constant variance across time</td>
      <td>H0: Homoskedasticity. White p=0.6486 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>✅ Stationary</td>
      <td>5.8489</td>
      <td>0.8278</td>
      <td>No volatility clustering detected (ARCH tests for clustered variance, not smooth changes)</td>
      <td>H0: No ARCH effects. ARCH p=0.8278 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>✅ Stationary</td>
      <td>0.0800</td>
      <td>1.0000</td>
      <td>No seasonal patterns detected</td>
      <td>H0: No seasonality. Ljung-Box p=1.0000 &gt;= 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>✅ Stationary</td>
      <td>0.1521</td>
      <td>1.0000</td>
      <td>No significant seasonal component detected</td>
      <td>H0: No seasonality. F-stat p=1.0000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



All tests pass - validates toolkit correctly identifies stationary data.

### 1.2 Noise + Baseline (Stationary)


```python
ts_baseline = pd.Series(
    100 + np.random.normal(0, 1, n),
    index=pd.date_range('2020-01-01', periods=n, freq='D')
)
plot_ts(ts_baseline.values, "Noise + Baseline (Stationary)")

result = toolkit.detect(ts_baseline, verbosity='detailed')
print(result.summary)
result.report(filepath='examples/case2_baseline_report.md')
```


    
![png](detailed_usage_files/detailed_usage_8_0.png)
    


    Trend Stationary: ✅ Yes
    Variance Stationary: ✅ Yes
    Seasonal Stationary: ✅ Yes





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>✅ Stationary</td>
      <td>-32.0477</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. ADF-c p=0.0000 &lt; 0.05, ADF-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>✅ Stationary</td>
      <td>0.1149</td>
      <td>0.1000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Stationary. KPSS-c p=0.1000 &gt; 0.05, KPSS-ct p=0.1000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>✅ Stationary</td>
      <td>-32.1132</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. PP-c p=0.0000 &lt; 0.05, PP-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-32.1820</td>
      <td>0.0000</td>
      <td>Stationary with structural breaks: level shift at obs 373, trend shift at obs 185, level+trend shift at obs 771. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0000, ZA-t p=0.0007, ZA-ct p=0.0002 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>0.2491</td>
      <td>0.8620</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments. Levene p=0.8620 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>0.6772</td>
      <td>0.8786</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.8786 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>✅ Stationary</td>
      <td>0.3723</td>
      <td>0.8302</td>
      <td>Constant variance across time</td>
      <td>H0: Homoskedasticity. White p=0.8302 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>✅ Stationary</td>
      <td>4.7745</td>
      <td>0.9057</td>
      <td>No volatility clustering detected (ARCH tests for clustered variance, not smooth changes)</td>
      <td>H0: No ARCH effects. ARCH p=0.9057 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>✅ Stationary</td>
      <td>0.0709</td>
      <td>1.0000</td>
      <td>No seasonal patterns detected</td>
      <td>H0: No seasonality. Ljung-Box p=1.0000 &gt;= 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>✅ Stationary</td>
      <td>0.1343</td>
      <td>1.0000</td>
      <td>No significant seasonal component detected</td>
      <td>H0: No seasonality. F-stat p=1.0000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



All tests pass - baseline shift doesn't affect stationarity.

### 1.3 Noise + Baseline + Trend (Non-Stationary)


```python
ts_trend = pd.Series(
    100 + 0.05 * np.arange(n) + np.random.normal(0, 1, n),
    index=pd.date_range('2020-01-01', periods=n, freq='D')
)
plot_ts(ts_trend.values, "Noise + Baseline + Trend (Non-Stationary)")

result = toolkit.detect(ts_trend, verbosity='detailed')
print(result.summary)
result.report(filepath='examples/case3_trend_report.md')
```


    
![png](detailed_usage_files/detailed_usage_11_0.png)
    


    Trend Stationary: ❌ No
    Variance Stationary: ❌ No
    Seasonal Stationary: ❌ No





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>❌ Non-stationary</td>
      <td>-0.3909</td>
      <td>0.9116</td>
      <td>Deterministic trend detected - stationary after detrending</td>
      <td>H0: Unit root. ADF-c p=0.9116 &gt;= 0.05, ADF-ct p=0.0000 &lt; 0.05. Deterministic trend.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>❌ Non-stationary</td>
      <td>5.1008</td>
      <td>0.0100</td>
      <td>Deterministic trend detected - stationary after detrending</td>
      <td>H0: Stationary. KPSS-c p=0.0100 &lt;= 0.05, KPSS-ct p=0.1000 &gt; 0.05. Deterministic trend.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>❌ Non-stationary</td>
      <td>-0.5899</td>
      <td>0.8732</td>
      <td>Deterministic trend detected - stationary after detrending</td>
      <td>H0: Unit root. PP-c p=0.8732 &gt;= 0.05, PP-ct p=0.0000 &lt; 0.05. Deterministic trend.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-32.1303</td>
      <td>0.0000</td>
      <td>Stationary with structural breaks: level shift at obs 743, trend shift at obs 305, level+trend shift at obs 743. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0000, ZA-t p=0.0007, ZA-ct p=0.0002 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>0.0976</td>
      <td>0.9614</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments. Levene p=0.9614 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>0.1229</td>
      <td>0.9890</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.9890 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>✅ Stationary</td>
      <td>0.8175</td>
      <td>0.6645</td>
      <td>Constant variance across time</td>
      <td>H0: Homoskedasticity. White p=0.6645 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>984.8041</td>
      <td>0.0000</td>
      <td>Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)</td>
      <td>H0: No ARCH effects. ARCH p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>❌ Non-stationary</td>
      <td>0.9923</td>
      <td>0.0000</td>
      <td>Seasonality detected (periods: 7, 12, 30, 52) - consider seasonal differencing (may trigger on trend/variance)</td>
      <td>H0: No seasonality. Ljung-Box p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>✅ Stationary</td>
      <td>0.1486</td>
      <td>1.0000</td>
      <td>No significant seasonal component detected</td>
      <td>H0: No seasonality. F-stat p=1.0000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



ADF, KPSS, PP correctly detect deterministic trend. ZA false positive (finds spurious breaks in smooth trend). ARCH fails due to trend-induced autocorrelation, not true volatility clustering. ACF detects trend as false seasonality - demonstrates cross-contamination between non-stationarity types.

### 1.4 Noise + Baseline + Seasonality (Non-Stationary)


```python
ts_seasonal = pd.Series(
    100 + 10 * np.sin(2 * np.pi * np.arange(n) / 52) + np.random.normal(0, 1, n),
    index=pd.date_range('2020-01-01', periods=n, freq='W')
)
plot_ts(ts_seasonal.values, "Noise + Baseline + Seasonality (Non-Stationary)")

result = toolkit.detect(ts_seasonal, verbosity='detailed')
print(result.summary)
result.report(filepath='examples/case4_seasonal_report.md')
```


    
![png](detailed_usage_files/detailed_usage_14_0.png)
    


    Trend Stationary: ✅ Yes
    Variance Stationary: ❌ No
    Seasonal Stationary: ❌ No





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>✅ Stationary</td>
      <td>-14.4895</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. ADF-c p=0.0000 &lt; 0.05, ADF-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>✅ Stationary</td>
      <td>0.0099</td>
      <td>0.1000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Stationary. KPSS-c p=0.1000 &gt; 0.05, KPSS-ct p=0.1000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>✅ Stationary</td>
      <td>-6.5116</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. PP-c p=0.0000 &lt; 0.05, PP-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-14.7602</td>
      <td>0.0000</td>
      <td>Stationary with structural breaks: level shift at obs 721, trend shift at obs 264, level+trend shift at obs 721. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0000, ZA-t p=0.0010, ZA-ct p=0.0007 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>0.4526</td>
      <td>0.7155</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments. Levene p=0.7155 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>0.3359</td>
      <td>0.9531</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.9531 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>✅ Stationary</td>
      <td>0.5885</td>
      <td>0.7451</td>
      <td>Constant variance across time</td>
      <td>H0: Homoskedasticity. White p=0.7451 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>962.2527</td>
      <td>0.0000</td>
      <td>Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)</td>
      <td>H0: No ARCH effects. ARCH p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>❌ Non-stationary</td>
      <td>0.9707</td>
      <td>0.0000</td>
      <td>Seasonality detected (periods: 52) - consider seasonal differencing (may trigger on trend/variance)</td>
      <td>H0: No seasonality. Ljung-Box p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>❌ Non-stationary</td>
      <td>62.4157</td>
      <td>0.0000</td>
      <td>Significant seasonal component detected (period 52) - consider seasonal differencing</td>
      <td>H0: No seasonality. F-stat p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



ACF and STL correctly detect 52-week seasonality. ARCH fails due to seasonal autocorrelation.

### 1.5 Noise + Baseline + Changing Variance (Non-Stationary)


```python
variance = 1 + 0.01 * np.arange(n)
ts_hetero = pd.Series(
    100 + np.random.normal(0, np.sqrt(variance), n),
    index=pd.date_range('2020-01-01', periods=n, freq='D')
)
plot_ts(ts_hetero.values, "Noise + Baseline + Changing Variance (Non-Stationary)")

result = toolkit.detect(ts_hetero, verbosity='detailed')
print(result.summary)
result.report(filepath='examples/case5_variance_report.md')
```


    
![png](detailed_usage_files/detailed_usage_17_0.png)
    


    Trend Stationary: ✅ Yes
    Variance Stationary: ❌ No
    Seasonal Stationary: ✅ Yes





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>✅ Stationary</td>
      <td>-31.8853</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. ADF-c p=0.0000 &lt; 0.05, ADF-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>✅ Stationary</td>
      <td>0.1248</td>
      <td>0.1000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Stationary. KPSS-c p=0.1000 &gt; 0.05, KPSS-ct p=0.1000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>✅ Stationary</td>
      <td>-31.9472</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. PP-c p=0.0000 &lt; 0.05, PP-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-31.9575</td>
      <td>0.0000</td>
      <td>Stationary with structural breaks: level shift at obs 848, trend shift at obs 835, level+trend shift at obs 777. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0000, ZA-t p=0.0007, ZA-ct p=0.0002 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>37.4542</td>
      <td>0.0000</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments. Levene p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>135.7226</td>
      <td>0.0000</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>87.0758</td>
      <td>0.0000</td>
      <td>Time-dependent variance detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Homoskedasticity. White p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>✅ Stationary</td>
      <td>11.1510</td>
      <td>0.3459</td>
      <td>No volatility clustering detected (ARCH tests for clustered variance, not smooth changes)</td>
      <td>H0: No ARCH effects. ARCH p=0.3459 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>✅ Stationary</td>
      <td>0.0894</td>
      <td>1.0000</td>
      <td>No seasonal patterns detected</td>
      <td>H0: No seasonality. Ljung-Box p=1.0000 &gt;= 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>✅ Stationary</td>
      <td>0.1473</td>
      <td>1.0000</td>
      <td>No significant seasonal component detected</td>
      <td>H0: No seasonality. F-stat p=1.0000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



Levene, Bartlett, White correctly detect growing variance. ARCH passes - validates it only detects clustered variance, not smooth changes.

### 1.6 Noise + Baseline + Clustered Variance (Non-Stationary)


```python
# Explicit volatility clustering (regime switches)
volatility = np.ones(n)
volatility[200:400] = 5  # High volatility period
volatility[600:800] = 0.2  # Low volatility period

ts_clustered = pd.Series(
    100 + np.random.normal(0, volatility, n),
    index=pd.date_range('2020-01-01', periods=n, freq='D')
)
plot_ts(ts_clustered.values, "Noise + Baseline + Clustered Variance (Non-Stationary)")

result = toolkit.detect(ts_clustered, verbosity='detailed')
print(result.summary)
result.report(filepath='examples/case6_clustered_report.md')
```


    
![png](detailed_usage_files/detailed_usage_20_0.png)
    


    Trend Stationary: ✅ Yes
    Variance Stationary: ❌ No
    Seasonal Stationary: ❌ No





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>✅ Stationary</td>
      <td>-5.9922</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. ADF-c p=0.0000 &lt; 0.05, ADF-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>✅ Stationary</td>
      <td>0.1188</td>
      <td>0.1000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Stationary. KPSS-c p=0.1000 &gt; 0.05, KPSS-ct p=0.1000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>✅ Stationary</td>
      <td>-31.4896</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. PP-c p=0.0000 &lt; 0.05, PP-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-7.3292</td>
      <td>0.0000</td>
      <td>Stationary with structural breaks: level shift at obs 306, trend shift at obs 359, level+trend shift at obs 306. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0000, ZA-t p=0.0031, ZA-ct p=0.0010 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>86.2640</td>
      <td>0.0000</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments. Levene p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>815.8958</td>
      <td>0.0000</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>52.7225</td>
      <td>0.0000</td>
      <td>Time-dependent variance detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Homoskedasticity. White p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>29.1246</td>
      <td>0.0012</td>
      <td>Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)</td>
      <td>H0: No ARCH effects. ARCH p=0.0012 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>❌ Non-stationary</td>
      <td>0.1372</td>
      <td>0.0000</td>
      <td>Seasonality detected (periods: 7, 12, 30, 52) - consider seasonal differencing (may trigger on trend/variance)</td>
      <td>H0: No seasonality. Ljung-Box p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>✅ Stationary</td>
      <td>0.2474</td>
      <td>1.0000</td>
      <td>No significant seasonal component detected</td>
      <td>H0: No seasonality. F-stat p=1.0000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



All variance tests correctly detect regime-switch volatility clustering. ARCH successfully identifies clustered variance. ACF false positive (detects variance as seasonality) - demonstrates test limitations.

## Part 2: A Simple Workflow

Now let's see how the toolkit guides preprocessing decisions. We'll create a time series with trend and 52-week seasonality - the kind of pattern you'd see in weekly business data. The workflow is straightforward: test it, transform it, test it again.

The initial test should identify both the trend and seasonality. We'll also see the ARCH test fail, but that's a false alarm - it's responding to the autocorrelation from trend and seasonality, not actual volatility clustering. This is one of those limitations we discovered in Part 1.

After differencing to remove both trend and seasonality, we retest. The trend and seasonality should be gone, but here's where it gets interesting: variance tests that passed before now fail. We didn't add any heteroscedasticity to the original data, yet it appeared after differencing. This is the toolkit's value - it reveals that transformations don't just solve problems, they can create new ones. Whether this new variance issue matters depends on your use case, but at least now you know it's there.


```python
# Generate data with trend and seasonality
ts_complex = pd.Series(
    100 + 0.05 * np.arange(n) + 10 * np.sin(2 * np.pi * np.arange(n) / 52) + np.random.normal(0, 1, n),
    index=pd.date_range('2020-01-01', periods=n, freq='W')
)
plot_ts(ts_complex.values, "Trend + Seasonality")
```


    
![png](detailed_usage_files/detailed_usage_23_0.png)
    


### Step 1: Initial Testing


```python
result_before = toolkit.detect(ts_complex, verbosity='detailed')
result_before.report()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>❌ Non-stationary</td>
      <td>-0.6090</td>
      <td>0.8689</td>
      <td>Deterministic trend detected - stationary after detrending</td>
      <td>H0: Unit root. ADF-c p=0.8689 &gt;= 0.05, ADF-ct p=0.0000 &lt; 0.05. Deterministic trend.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>❌ Non-stationary</td>
      <td>4.4629</td>
      <td>0.0100</td>
      <td>Deterministic trend detected - stationary after detrending</td>
      <td>H0: Stationary. KPSS-c p=0.0100 &lt;= 0.05, KPSS-ct p=0.1000 &gt; 0.05. Deterministic trend.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>❌ Non-stationary</td>
      <td>-2.7384</td>
      <td>0.0676</td>
      <td>Deterministic trend detected - stationary after detrending</td>
      <td>H0: Unit root. PP-c p=0.0676 &gt;= 0.05, PP-ct p=0.0000 &lt; 0.05. Deterministic trend.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-16.0353</td>
      <td>0.0000</td>
      <td>Stationary with structural breaks: level shift at obs 407, trend shift at obs 849, level+trend shift at obs 824. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0000, ZA-t p=0.0010, ZA-ct p=0.0007 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>0.8956</td>
      <td>0.4429</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments. Levene p=0.4429 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>1.9897</td>
      <td>0.5745</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.5745 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>✅ Stationary</td>
      <td>0.1837</td>
      <td>0.9122</td>
      <td>Constant variance across time</td>
      <td>H0: Homoskedasticity. White p=0.9122 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>982.2571</td>
      <td>0.0000</td>
      <td>Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)</td>
      <td>H0: No ARCH effects. ARCH p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>❌ Non-stationary</td>
      <td>0.9907</td>
      <td>0.0000</td>
      <td>Seasonality detected (periods: 52) - consider seasonal differencing (may trigger on trend/variance)</td>
      <td>H0: No seasonality. Ljung-Box p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>❌ Non-stationary</td>
      <td>59.1103</td>
      <td>0.0000</td>
      <td>Significant seasonal component detected (period 52) - consider seasonal differencing</td>
      <td>H0: No seasonality. F-stat p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



The toolkit correctly identifies the non-stationarity. ADF, KPSS, and PP flag the deterministic trend. ZA shows a false positive (finds spurious breaks in smooth trends). Variance tests correctly pass except ARCH, which fails due to trend/seasonality-induced autocorrelation rather than true volatility clustering. Both seasonal tests correctly detect the 52-week pattern. These results validate the toolkit's detection capabilities.

### Step 2: Apply Transformations


```python
# Detrend
ts_detrended = ts_complex.diff().dropna()
plot_ts(ts_detrended.values, "After Detrending")
```


    
![png](detailed_usage_files/detailed_usage_28_0.png)
    


### Step 3: Retest


```python
result_after = toolkit.detect(ts_detrended, verbosity='detailed')
result_after.report()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>✅ Stationary</td>
      <td>-17.0140</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. ADF-c p=0.0000 &lt; 0.05, ADF-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>✅ Stationary</td>
      <td>0.0094</td>
      <td>0.1000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Stationary. KPSS-c p=0.1000 &gt; 0.05, KPSS-ct p=0.1000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>✅ Stationary</td>
      <td>-40.4184</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. PP-c p=0.0000 &lt; 0.05, PP-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-17.0037</td>
      <td>0.0000</td>
      <td>Stationary with structural breaks: level shift at obs 737, trend shift at obs 820, level+trend shift at obs 796. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0000, ZA-t p=0.0010, ZA-ct p=0.0007 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>0.3019</td>
      <td>0.8241</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments. Levene p=0.8241 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>0.6646</td>
      <td>0.8815</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.8815 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>✅ Stationary</td>
      <td>0.3870</td>
      <td>0.8241</td>
      <td>Constant variance across time</td>
      <td>H0: Homoskedasticity. White p=0.8241 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>20.6880</td>
      <td>0.0234</td>
      <td>Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)</td>
      <td>H0: No ARCH effects. ARCH p=0.0234 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>❌ Non-stationary</td>
      <td>0.2717</td>
      <td>0.0000</td>
      <td>Seasonality detected (periods: 52) - consider seasonal differencing (may trigger on trend/variance)</td>
      <td>H0: No seasonality. Ljung-Box p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>✅ Stationary</td>
      <td>0.6043</td>
      <td>1.0000</td>
      <td>No significant seasonal component detected</td>
      <td>H0: No seasonality. F-stat p=1.0000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



The transformations successfully addressed the trend - all trend tests now pass. STL confirm the seasonal pattern is eliminated, though ACF still flags residual autocorrelation albeit with smaller test statistic 0.99 prior vs 0.27 now. Variance tests continue to pass except ARCH likely due to its tendency to trigger on temporal patterns.

### Step 4: Would Seasonal Differencing Work ?


```python
# Seasonal differencing
ts_transformed = ts_detrended.diff(52).dropna()
plot_ts(ts_transformed.values, "After Detrending + Seasonal Differencing")
```


    
![png](detailed_usage_files/detailed_usage_33_0.png)
    



```python
result_after = toolkit.detect(ts_transformed, verbosity='detailed')
result_after.report()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>✅ Stationary</td>
      <td>-12.4676</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. ADF-c p=0.0000 &lt; 0.05, ADF-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>✅ Stationary</td>
      <td>0.0397</td>
      <td>0.1000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Stationary. KPSS-c p=0.1000 &gt; 0.05, KPSS-ct p=0.1000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>✅ Stationary</td>
      <td>-154.6462</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. PP-c p=0.0000 &lt; 0.05, PP-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-12.5058</td>
      <td>0.0000</td>
      <td>Stationary with structural breaks: level shift at obs 447, trend shift at obs 804, level+trend shift at obs 767. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0000, ZA-t p=0.0013, ZA-ct p=0.0008 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>0.3113</td>
      <td>0.8172</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments. Levene p=0.8172 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>1.9347</td>
      <td>0.5861</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.5861 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>✅ Stationary</td>
      <td>0.1408</td>
      <td>0.9320</td>
      <td>Constant variance across time</td>
      <td>H0: Homoskedasticity. White p=0.9320 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>61.3137</td>
      <td>0.0000</td>
      <td>Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)</td>
      <td>H0: No ARCH effects. ARCH p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>❌ Non-stationary</td>
      <td>0.5111</td>
      <td>0.0000</td>
      <td>Seasonality detected (periods: 52) - consider seasonal differencing (may trigger on trend/variance)</td>
      <td>H0: No seasonality. Ljung-Box p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>✅ Stationary</td>
      <td>0.0249</td>
      <td>1.0000</td>
      <td>No significant seasonal component detected</td>
      <td>H0: No seasonality. F-stat p=1.0000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



After seasonal differencing, the ACF outcome got worse indicating just conventional 52 lag differencing won't work to handle it. Perhaps, more detailed analysis needed. But that's besides the point that the toolkit does its job in letting you know what's happenning.

## Part 3: When Things Get Complicated

Real data rarely has just one problem. This series combines trend, seasonality, and heteroscedasticity - all three at once. Now the question isn't just "what transformations do I need?" but "in what order?"

We'll follow the recommended sequence: stabilize variance first, then remove trend, then handle seasonality. But here's what makes this interesting - we test after every single step. Not because we're being thorough, but because transformations interact in ways you can't predict.

Watch what happens: Box-Cox stabilizes the variance nicely. Then we difference to remove the trend, and suddenly all the variance tests fail again. The stabilization didn't survive the differencing. Meanwhile, ACF keeps flagging seasonality even after STL say it's gone. Is there really seasonality left, or is ACF just picking up the variance issues? The only way to know is to test at each step and see what's actually happening versus what you expected to happen.


```python
# Generate complex data
variance = 1 + 0.05 * np.arange(n)
ts_full = pd.Series(
    100 + 0.05 * np.arange(n) + 10 * np.sin(2 * np.pi * np.arange(n) / 52) + np.random.normal(0, np.sqrt(variance), n),
    index=pd.date_range('2020-01-01', periods=n, freq='W')
)
plot_ts(ts_full.values, "Trend + Seasonality + Heteroscedasticity")
```


    
![png](detailed_usage_files/detailed_usage_37_0.png)
    



```python
# Test
result_full_before = toolkit.detect(ts_full, verbosity='detailed')
result_full_before.report()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>❌ Non-stationary</td>
      <td>-1.9483</td>
      <td>0.3097</td>
      <td>Deterministic trend detected - stationary after detrending</td>
      <td>H0: Unit root. ADF-c p=0.3097 &gt;= 0.05, ADF-ct p=0.0000 &lt; 0.05. Deterministic trend.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>❌ Non-stationary</td>
      <td>4.4473</td>
      <td>0.0100</td>
      <td>Deterministic trend detected - stationary after detrending</td>
      <td>H0: Stationary. KPSS-c p=0.0100 &lt;= 0.05, KPSS-ct p=0.1000 &gt; 0.05. Deterministic trend.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>✅ Stationary</td>
      <td>-7.5072</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. PP-c p=0.0000 &lt; 0.05, PP-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-13.5080</td>
      <td>0.0000</td>
      <td>Stationary with structural breaks: level shift at obs 443, trend shift at obs 612, level+trend shift at obs 443. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0000, ZA-t p=0.0011, ZA-ct p=0.0008 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>4.1204</td>
      <td>0.0064</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments. Levene p=0.0064 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>18.9937</td>
      <td>0.0003</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.0003 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>16.7963</td>
      <td>0.0002</td>
      <td>Time-dependent variance detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Homoskedasticity. White p=0.0002 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>848.2545</td>
      <td>0.0000</td>
      <td>Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)</td>
      <td>H0: No ARCH effects. ARCH p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>❌ Non-stationary</td>
      <td>0.8948</td>
      <td>0.0000</td>
      <td>Seasonality detected (periods: 52) - consider seasonal differencing (may trigger on trend/variance)</td>
      <td>H0: No seasonality. Ljung-Box p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>❌ Non-stationary</td>
      <td>2.6187</td>
      <td>0.0000</td>
      <td>Significant seasonal component detected (period 52) - consider seasonal differencing</td>
      <td>H0: No seasonality. F-stat p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



This complex series combines all three non-stationarity types. The trend tests show interesting behavior: ADF and KPSS correctly flag the deterministic trend, while PP unexpectedly passes. PP's non-parametric variance correction (Newey-West estimator) appears to break down with long-period seasonal autocorrelation (lag 52) plus growing variance. ZA finds spurious structural breaks in the smooth trend.

Variance tests: Sometimes, Levene barely passes (such as with p=0.059), but mostly all variance tests detect the 50-fold variance growth. ARCH fails, likely from trend/seasonality autocorrelation rather than true clustering.

Seasonality: Both ACF and STL detect the 52-period cycle, confirming the presence of deterministic (fixed, repeating) seasonal patterns.

The toolkit reveals each layer of non-stationarity and how different tests respond to combined effects.

### Step 1: Stabilize Variance (Box-Cox)


```python
from scipy.stats import boxcox
ts_boxcox, lambda_param = boxcox(ts_full)
ts_boxcox = pd.Series(ts_boxcox, index=ts_full.index)
plot_ts(ts_boxcox.values, f"After Box-Cox (λ={lambda_param:.3f})")

result_step1 = toolkit.detect(ts_boxcox, verbosity='detailed')
result_step1.report()
```


    
![png](detailed_usage_files/detailed_usage_41_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>❌ Non-stationary</td>
      <td>-1.9987</td>
      <td>0.2871</td>
      <td>Deterministic trend detected - stationary after detrending</td>
      <td>H0: Unit root. ADF-c p=0.2871 &gt;= 0.05, ADF-ct p=0.0000 &lt; 0.05. Deterministic trend.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>❌ Non-stationary</td>
      <td>4.4279</td>
      <td>0.0100</td>
      <td>Deterministic trend detected - stationary after detrending</td>
      <td>H0: Stationary. KPSS-c p=0.0100 &lt;= 0.05, KPSS-ct p=0.1000 &gt; 0.05. Deterministic trend.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>✅ Stationary</td>
      <td>-7.2638</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. PP-c p=0.0000 &lt; 0.05, PP-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-13.4272</td>
      <td>0.0000</td>
      <td>Stationary with structural breaks: level shift at obs 164, trend shift at obs 278, level+trend shift at obs 443. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0000, ZA-t p=0.0011, ZA-ct p=0.0008 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>1.2255</td>
      <td>0.2992</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments. Levene p=0.2992 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>4.5495</td>
      <td>0.2079</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.2079 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>✅ Stationary</td>
      <td>0.3405</td>
      <td>0.8435</td>
      <td>Constant variance across time</td>
      <td>H0: Homoskedasticity. White p=0.8435 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>862.4914</td>
      <td>0.0000</td>
      <td>Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)</td>
      <td>H0: No ARCH effects. ARCH p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>❌ Non-stationary</td>
      <td>0.9003</td>
      <td>0.0000</td>
      <td>Seasonality detected (periods: 52) - consider seasonal differencing (may trigger on trend/variance)</td>
      <td>H0: No seasonality. Ljung-Box p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>❌ Non-stationary</td>
      <td>2.8080</td>
      <td>0.0000</td>
      <td>Significant seasonal component detected (period 52) - consider seasonal differencing</td>
      <td>H0: No seasonality. F-stat p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



Box-Cox successfully stabilized the variance - Levene, Bartlett, and White all pass. The ARCH test still fails, but as its notes indicate, this can occur due to autocorrelation from trend and seasonality. The trend and seasonal patterns remain unchanged, as expected after variance-only transformation.

### Step 2: Remove Trend (Differencing)


```python
ts_diff = ts_boxcox.diff().dropna()
plot_ts(ts_diff.values, "After Box-Cox + Differencing")

result_step2 = toolkit.detect(ts_diff, verbosity='detailed')
result_step2.report()
```


    
![png](detailed_usage_files/detailed_usage_44_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>✅ Stationary</td>
      <td>-10.6874</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. ADF-c p=0.0000 &lt; 0.05, ADF-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>✅ Stationary</td>
      <td>0.1370</td>
      <td>0.1000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Stationary. KPSS-c p=0.1000 &gt; 0.05, KPSS-ct p=0.1000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>✅ Stationary</td>
      <td>-64.3445</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. PP-c p=0.0000 &lt; 0.05, PP-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-10.6992</td>
      <td>0.0000</td>
      <td>Stationary with structural breaks: level shift at obs 440, trend shift at obs 818, level+trend shift at obs 807. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0000, ZA-t p=0.0017, ZA-ct p=0.0009 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>24.2371</td>
      <td>0.0000</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments. Levene p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>89.4104</td>
      <td>0.0000</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>59.7366</td>
      <td>0.0000</td>
      <td>Time-dependent variance detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Homoskedasticity. White p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>106.9324</td>
      <td>0.0000</td>
      <td>Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)</td>
      <td>H0: No ARCH effects. ARCH p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>❌ Non-stationary</td>
      <td>0.4837</td>
      <td>0.0000</td>
      <td>Seasonality detected (periods: 52) - consider seasonal differencing (may trigger on trend/variance)</td>
      <td>H0: No seasonality. Ljung-Box p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>✅ Stationary</td>
      <td>0.2403</td>
      <td>1.0000</td>
      <td>No significant seasonal component detected</td>
      <td>H0: No seasonality. F-stat p=1.0000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



Differencing removed the trend - all trend tests now pass. However, all variance tests now fail. The toolkit reveals that the variance stabilization from Step 1 didn't survive the differencing transformation. STL pass for seasonality, but ACF still fails. Given ACF's note that it "may trigger on trend/variance," and the clear variance issues, the ACF signal is likely variance-related rather than true seasonality.

### Step 3: Remove Seasonality (Seasonal Differencing)


```python
ts_full_transformed = ts_diff.diff(52).dropna()
plot_ts(ts_full_transformed.values, "After Box-Cox + Differencing + Seasonal Differencing")

result_step3 = toolkit.detect(ts_full_transformed, verbosity='detailed')
result_step3.report()
```


    
![png](detailed_usage_files/detailed_usage_47_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>✅ Stationary</td>
      <td>-11.6336</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. ADF-c p=0.0000 &lt; 0.05, ADF-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>✅ Stationary</td>
      <td>0.1494</td>
      <td>0.1000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Stationary. KPSS-c p=0.1000 &gt; 0.05, KPSS-ct p=0.1000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>✅ Stationary</td>
      <td>-158.3697</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. PP-c p=0.0000 &lt; 0.05, PP-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-11.6565</td>
      <td>0.0000</td>
      <td>Stationary with structural breaks: level shift at obs 751, trend shift at obs 746, level+trend shift at obs 612. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0000, ZA-t p=0.0015, ZA-ct p=0.0008 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>25.5875</td>
      <td>0.0000</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments. Levene p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>100.2125</td>
      <td>0.0000</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>67.9330</td>
      <td>0.0000</td>
      <td>Time-dependent variance detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Homoskedasticity. White p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>122.7720</td>
      <td>0.0000</td>
      <td>Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)</td>
      <td>H0: No ARCH effects. ARCH p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>❌ Non-stationary</td>
      <td>0.5145</td>
      <td>0.0000</td>
      <td>Seasonality detected (periods: 52) - consider seasonal differencing (may trigger on trend/variance)</td>
      <td>H0: No seasonality. Ljung-Box p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>✅ Stationary</td>
      <td>0.0182</td>
      <td>1.0000</td>
      <td>No significant seasonal component detected</td>
      <td>H0: No seasonality. F-stat p=1.0000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



Seasonal differencing applied, but the results are largely unchanged from Step 2. Variance tests still fail, ACF still fails, while STL still passes. This confirms what we suspected after Step 2: the ACF signal was variance-related, not true seasonality. The toolkit helped us avoid unnecessary transformation by revealing the true source of the ACF failure through iterative testing.

## Part 4: Real Data Doesn't Read the Textbook

Everything we've done so far used synthetic data where we controlled exactly what non-stationarity existed. Real data is messier. This Walmart sales series shows mixed signals right from the start - KPSS says there's a trend, but ADF, PP, and ZA all say the series is stationary. Who's right? Probably all of them - the trend is weak enough that it's borderline.

But here's the real lesson: the same transformations we used in Part 3 produce completely different results here. In Part 3, Box-Cox stabilized variance but differencing destabilized it. Here? Box-Cox helps a little, differencing helps more, then seasonal differencing reverses some of the gains. Same transformations, different data, different outcomes.

This is why you can't just follow a recipe. You can't assume "Box-Cox then difference" will always work the same way. The toolkit lets you see what actually happened with your specific data, so you can decide whether to keep going or try a different approach.


```python
# Load data
df = pd.read_csv('examples/walmartSales_FOODS_3_586_TX_1_evaluation.csv')
df['date'] = pd.to_datetime(df['date'])
ts_walmart = pd.Series(df['value'].values, index=pd.DatetimeIndex(df['date'], freq='W-MON'))
plot_ts(ts_walmart.values, "Walmart Sales Data")
```


    
![png](detailed_usage_files/detailed_usage_50_0.png)
    



```python
# Test
result_walmart = toolkit.detect(ts_walmart, verbosity='detailed')
print(result_walmart.summary)
result_walmart.report(filepath='examples/walmart_report.md')
```

    Trend Stationary: ❌ No
    Variance Stationary: ❌ No
    Seasonal Stationary: ❌ No





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>✅ Stationary</td>
      <td>-3.2863</td>
      <td>0.0155</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. ADF-c p=0.0155 &lt; 0.05, ADF-ct p=0.0039 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>❌ Non-stationary</td>
      <td>1.1964</td>
      <td>0.0100</td>
      <td>Deterministic trend detected - stationary after detrending</td>
      <td>H0: Stationary. KPSS-c p=0.0100 &lt;= 0.05, KPSS-ct p=0.0951 &gt; 0.05. Deterministic trend.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>✅ Stationary</td>
      <td>-7.8514</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. PP-c p=0.0000 &lt; 0.05, PP-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-5.5103</td>
      <td>0.0040</td>
      <td>Stationary with structural breaks: level shift at obs 99, trend shift at obs 155, level+trend shift at obs 99. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0040, ZA-t p=0.0285, ZA-ct p=0.0157 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>5.7629</td>
      <td>0.0008</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments. Levene p=0.0008 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>30.8232</td>
      <td>0.0000</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>10.8138</td>
      <td>0.0045</td>
      <td>Time-dependent variance detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Homoskedasticity. White p=0.0045 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>173.6824</td>
      <td>0.0000</td>
      <td>Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)</td>
      <td>H0: No ARCH effects. ARCH p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>❌ Non-stationary</td>
      <td>0.7508</td>
      <td>0.0000</td>
      <td>Seasonality detected (periods: 52) - consider seasonal differencing (may trigger on trend/variance)</td>
      <td>H0: No seasonality. Ljung-Box p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>❌ Non-stationary</td>
      <td>2.3183</td>
      <td>0.0000</td>
      <td>Significant seasonal component detected (period 52) - consider seasonal differencing</td>
      <td>H0: No seasonality. F-stat p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



The Walmart sales data presents an interesting case of mixed signals. For trend, KPSS flags a deterministic trend while ADF, PP, and ZA all pass. This disagreement suggests the trend is relatively weak or the series is borderline stationary around a constant mean. The majority of tests indicate stationarity, though the KPSS result suggests some trend-like behavior may be present.

The variance tests tell a clear story - all four tests fail, indicating heteroscedasticity. The variance changes across time segments (Levene, Bartlett), shows time-dependent patterns (White), and exhibits clustering (ARCH, though likely due to autocorrelation from trend/seasonality). Box-Cox transformation would address this.

For seasonality, both ACF and STL detect the strong 52-week pattern, confirming the presence of deterministic seasonal patterns.

This real-world data demonstrates how multiple forms of non-stationarity coexist, and how the toolkit's comprehensive testing reveals each layer. The next step would be to apply transformations iteratively and retest to see how each transformation affects the overall stationarity profile.

### Step 1: Stabilize Variance (Box-Cox)


```python
from scipy.stats import boxcox
ts_walmart_boxcox, lambda_walmart = boxcox(ts_walmart)
ts_walmart_boxcox = pd.Series(ts_walmart_boxcox, index=ts_walmart.index)
plot_ts(ts_walmart_boxcox.values, f"Walmart - After Box-Cox (λ={lambda_walmart:.3f})")

result_walmart_step1 = toolkit.detect(ts_walmart_boxcox, verbosity='detailed')
result_walmart_step1.report()
```


    
![png](detailed_usage_files/detailed_usage_54_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>✅ Stationary</td>
      <td>-3.3206</td>
      <td>0.0140</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. ADF-c p=0.0140 &lt; 0.05, ADF-ct p=0.0035 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>❌ Non-stationary</td>
      <td>1.1708</td>
      <td>0.0100</td>
      <td>Deterministic trend detected - stationary after detrending</td>
      <td>H0: Stationary. KPSS-c p=0.0100 &lt;= 0.05, KPSS-ct p=0.0932 &gt; 0.05. Deterministic trend.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>✅ Stationary</td>
      <td>-8.4822</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. PP-c p=0.0000 &lt; 0.05, PP-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-5.5447</td>
      <td>0.0036</td>
      <td>Stationary with structural breaks: level shift at obs 99, trend shift at obs 155, level+trend shift at obs 99. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0036, ZA-t p=0.0245, ZA-ct p=0.0142 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>4.4248</td>
      <td>0.0047</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments. Levene p=0.0047 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>27.2799</td>
      <td>0.0000</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>7.6245</td>
      <td>0.0221</td>
      <td>Time-dependent variance detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Homoskedasticity. White p=0.0221 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>172.4148</td>
      <td>0.0000</td>
      <td>Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)</td>
      <td>H0: No ARCH effects. ARCH p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>❌ Non-stationary</td>
      <td>0.7297</td>
      <td>0.0000</td>
      <td>Seasonality detected (periods: 52) - consider seasonal differencing (may trigger on trend/variance)</td>
      <td>H0: No seasonality. Ljung-Box p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>❌ Non-stationary</td>
      <td>2.1488</td>
      <td>0.0000</td>
      <td>Significant seasonal component detected (period 52) - consider seasonal differencing</td>
      <td>H0: No seasonality. F-stat p=0.0000 &lt;= 0.05. Reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



Box-Cox reduced but didn't eliminate the variance non-stationarity. The p-values improved (Levene 0.0047 vs 0.0008, White 0.0221 vs 0.0045), but all variance tests still fail. This shows Box-Cox helped but the heteroscedasticity persists. The trend and seasonal patterns remain unchanged, as expected.

### Step 2: Remove Trend (Differencing)


```python
ts_walmart_diff = ts_walmart_boxcox.diff().dropna()
plot_ts(ts_walmart_diff.values, "Walmart - After Box-Cox + Differencing")

result_walmart_step2 = toolkit.detect(ts_walmart_diff, verbosity='detailed')
result_walmart_step2.report()
```


    
![png](detailed_usage_files/detailed_usage_57_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>✅ Stationary</td>
      <td>-9.6556</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. ADF-c p=0.0000 &lt; 0.05, ADF-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>✅ Stationary</td>
      <td>0.1829</td>
      <td>0.1000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Stationary. KPSS-c p=0.1000 &gt; 0.05, KPSS-ct p=0.1000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>✅ Stationary</td>
      <td>-30.6166</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. PP-c p=0.0000 &lt; 0.05, PP-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-9.7740</td>
      <td>0.0000</td>
      <td>Stationary with structural breaks: level shift at obs 156, trend shift at obs 41, level+trend shift at obs 51. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0000, ZA-t p=0.0019, ZA-ct p=0.0009 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>1.7370</td>
      <td>0.1597</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments. Levene p=0.1597 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>10.7914</td>
      <td>0.0129</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.0129 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>✅ Stationary</td>
      <td>2.1175</td>
      <td>0.3469</td>
      <td>Constant variance across time</td>
      <td>H0: Homoskedasticity. White p=0.3469 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>21.4123</td>
      <td>0.0184</td>
      <td>Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)</td>
      <td>H0: No ARCH effects. ARCH p=0.0184 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>❌ Non-stationary</td>
      <td>0.2904</td>
      <td>0.0003</td>
      <td>Seasonality detected (periods: 52) - consider seasonal differencing (may trigger on trend/variance)</td>
      <td>H0: No seasonality. Ljung-Box p=0.0003 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>✅ Stationary</td>
      <td>0.7440</td>
      <td>0.9929</td>
      <td>No significant seasonal component detected</td>
      <td>H0: No seasonality. F-stat p=0.9929 &gt; 0.05. Fail to reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



Differencing removed the trend and unexpectedly improved variance stationarity - Levene and White now pass, though Bartlett and ARCH still fail. This contrasts with Part 3, where differencing worsened variance. The difference: Part 3 had synthetic growing variance that Box-Cox stabilized but differencing destabilized; here, real-world heteroscedasticity responds differently to the same transformations.

For seasonality, STL now passes, but ACF still detects patterns. Given ACF's note about triggering on variance, and with Bartlett/ARCH still failing, the ACF signal is likely variance-related rather than true seasonality. This demonstrates the toolkit's value - revealing how transformations affect different forms of non-stationarity in unpredictable ways, enabling informed decisions about whether additional steps are needed.

### Step 3: Remove Seasonality (Seasonal Differencing)


```python
ts_walmart_seasonal = ts_walmart_diff.diff(52).dropna()
plot_ts(ts_walmart_seasonal.values, "Walmart - After Box-Cox + Differencing + Seasonal Differencing")

result_walmart_step3 = toolkit.detect(ts_walmart_seasonal, verbosity='detailed')
result_walmart_step3.report()
```


    
![png](detailed_usage_files/detailed_usage_60_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test Type</th>
      <th>Test</th>
      <th>Result</th>
      <th>Statistic</th>
      <th>P-value</th>
      <th>Note</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trend</td>
      <td>Augmented Dickey-Fuller (ADF) Test</td>
      <td>✅ Stationary</td>
      <td>-5.2695</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. ADF-c p=0.0000 &lt; 0.05, ADF-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trend</td>
      <td>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</td>
      <td>✅ Stationary</td>
      <td>0.2918</td>
      <td>0.1000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Stationary. KPSS-c p=0.1000 &gt; 0.05, KPSS-ct p=0.0939 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trend</td>
      <td>Phillips-Perron (PP) Test</td>
      <td>✅ Stationary</td>
      <td>-37.6395</td>
      <td>0.0000</td>
      <td>Stationary around constant mean</td>
      <td>H0: Unit root. PP-c p=0.0000 &lt; 0.05, PP-ct p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trend</td>
      <td>Zivot-Andrews Test for Structural Breaks</td>
      <td>✅ Stationary</td>
      <td>-5.7648</td>
      <td>0.0014</td>
      <td>Stationary with structural breaks: level shift at obs 140, trend shift at obs 111, level+trend shift at obs 140. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.</td>
      <td>H0: Unit root with no break. ZA-c p=0.0014, ZA-t p=0.0050, ZA-ct p=0.0051 &lt; 0.05. Reject H0 - stationary.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Variance</td>
      <td>Levene's Test for Variance Homogeneity</td>
      <td>✅ Stationary</td>
      <td>2.3456</td>
      <td>0.0738</td>
      <td>Constant variance across time</td>
      <td>H0: Equal variances across segments. Levene p=0.0738 &gt; 0.05. Fail to reject H0.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Variance</td>
      <td>Bartlett's Test for Variance Homogeneity</td>
      <td>❌ Non-stationary</td>
      <td>20.4189</td>
      <td>0.0001</td>
      <td>Variance changes detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Equal variances across segments (assumes normality). Bartlett p=0.0001 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Variance</td>
      <td>White's Test for Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>6.2282</td>
      <td>0.0444</td>
      <td>Time-dependent variance detected - consider Box-Cox or Yeo-Johnson transform</td>
      <td>H0: Homoskedasticity. White p=0.0444 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Variance</td>
      <td>ARCH Test for Conditional Heteroskedasticity</td>
      <td>❌ Non-stationary</td>
      <td>18.5884</td>
      <td>0.0458</td>
      <td>Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)</td>
      <td>H0: No ARCH effects. ARCH p=0.0458 &lt;= 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Seasonal</td>
      <td>ACF/PACF Peak Detection</td>
      <td>❌ Non-stationary</td>
      <td>0.4514</td>
      <td>0.0000</td>
      <td>Seasonality detected (periods: 52) - consider seasonal differencing (may trigger on trend/variance)</td>
      <td>H0: No seasonality. Ljung-Box p=0.0000 &lt; 0.05. Reject H0.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Seasonal</td>
      <td>STL Decomposition</td>
      <td>✅ Stationary</td>
      <td>0.1897</td>
      <td>1.0000</td>
      <td>No significant seasonal component detected</td>
      <td>H0: No seasonality. F-stat p=1.0000 &gt; 0.05. Fail to reject H0.</td>
    </tr>
  </tbody>
</table>
</div>



Seasonal differencing worsened variance stationarity - White now fails again (was passing), and Levene is borderline (p=0.0738). Only Levene passes, while Bartlett, White, and ARCH all fail. For seasonality, STL still passes, but ACF still fails. The ACF signal persists despite STL showing no seasonal component, confirming it's detecting autocorrelation from the variance issues rather than true seasonality.

Box-Cox helped variance, differencing helped more, but seasonal differencing reversed some gains. Same transformations as Part 3, completely different trajectory.

## Conclusion

The toolkit's value isn't in prescribing solutions - it's in revealing what's actually there. Single-test approaches miss the nuances we've seen throughout this notebook: mixed signals between tests, false positives from test limitations, and cross-contamination where one type of non-stationarity triggers tests designed for another type.

More importantly, assumptions about transformations can fail on real data. Part 3 and Part 4 used identical transformations but produced opposite variance outcomes. You can't predict these interactions - you have to test and see. The iterative test-transform-retest workflow reveals the truth at each step, letting you make informed decisions for your specific data and use case.
