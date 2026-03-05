# Stationarity Detection Report

## Summary

- Trend Stationary: ✅ Yes
- Variance Stationary: ❌ No
- Seasonal Stationary: ❌ No

## Trend Tests

All tests passed ✅

## Variance Tests

### Levene's Test for Variance Homogeneity

- Result: ✅ Stationary
- Note: Constant variance across time
- Interpretation: H0: Equal variances across segments. Levene p=0.7155 > 0.05. Fail to reject H0.
- Statistic: 0.4526
- P-value: 0.7155

### Bartlett's Test for Variance Homogeneity

- Result: ✅ Stationary
- Note: Constant variance across time
- Interpretation: H0: Equal variances across segments (assumes normality). Bartlett p=0.9531 > 0.05. Fail to reject H0.
- Statistic: 0.3359
- P-value: 0.9531

### White's Test for Heteroskedasticity

- Result: ✅ Stationary
- Note: Constant variance across time
- Interpretation: H0: Homoskedasticity. White p=0.7451 > 0.05. Fail to reject H0.
- Statistic: 0.5885
- P-value: 0.7451

### ARCH Test for Conditional Heteroskedasticity

- Result: ❌ Non-stationary
- Note: Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)
- Interpretation: H0: No ARCH effects. ARCH p=0.0000 <= 0.05. Reject H0.
- Statistic: 962.2527
- P-value: 0.0000

## Seasonal Tests

### ACF/PACF Peak Detection

- Result: ❌ Non-stationary
- Note: Seasonality detected (periods: 52) - consider seasonal differencing (may trigger on trend/variance)
- Interpretation: H0: No seasonality. Ljung-Box p=0.0000 < 0.05. Reject H0.
- Statistic: 0.9707
- P-value: 0.0000

### STL Decomposition

- Result: ❌ Non-stationary
- Note: Significant seasonal component detected (period 52) - consider seasonal differencing
- Interpretation: H0: No seasonality. F-stat p=0.0000 <= 0.05. Reject H0.
- Statistic: 62.4157
- P-value: 0.0000

### OCSB

- Result: ✅ Stationary
- Note: No seasonal unit roots detected (period 52). OCSB detects random-walk seasonality, not fixed patterns - use STL for deterministic seasonality
- Interpretation: H0: Seasonal unit root present. OCSB t-stat=-3.2532 < -2.8600. Reject H0.
- Statistic: -3.2532
- P-value: 0.0250
