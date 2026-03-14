# Stationarity Detection Report

## Summary

- Trend Stationary: ✅ Yes
- Variance Stationary: ❌ No
- Seasonal Stationary: ❌ No

## Trend Tests

All tests passed ✅

## Variance Tests

### Levene's Test for Variance Homogeneity

- Result: ❌ Non-stationary
- Note: Variance changes detected - consider Box-Cox or Yeo-Johnson transform
- Interpretation: H0: Equal variances across segments. Levene p=0.0000 <= 0.05. Reject H0.
- Statistic: 87.7480
- P-value: 0.0000

### Bartlett's Test for Variance Homogeneity

- Result: ❌ Non-stationary
- Note: Variance changes detected - consider Box-Cox or Yeo-Johnson transform
- Interpretation: H0: Equal variances across segments (assumes normality). Bartlett p=0.0000 <= 0.05. Reject H0.
- Statistic: 772.2239
- P-value: 0.0000

### White's Test for Heteroskedasticity

- Result: ❌ Non-stationary
- Note: Time-dependent variance detected - consider Box-Cox or Yeo-Johnson transform
- Interpretation: H0: Homoskedasticity. White p=0.0000 <= 0.05. Reject H0.
- Statistic: 55.9629
- P-value: 0.0000

### ARCH Test for Conditional Heteroskedasticity

- Result: ❌ Non-stationary
- Note: Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)
- Interpretation: H0: No ARCH effects. ARCH p=0.0000 <= 0.05. Reject H0.
- Statistic: 47.1623
- P-value: 0.0000

## Seasonal Tests

### ACF/PACF Peak Detection

- Result: ❌ Non-stationary
- Note: Seasonality detected (periods: 7, 12, 30, 52) - consider seasonal differencing (may trigger on trend/variance)
- Interpretation: H0: No seasonality. Ljung-Box p=0.0000 < 0.05. Reject H0.
- Statistic: 0.1650
- P-value: 0.0000

### STL Decomposition

- Result: ✅ Stationary
- Note: No significant seasonal component detected
- Interpretation: H0: No seasonality. F-stat p=1.0000 > 0.05. Fail to reject H0.
- Statistic: 0.1081
- P-value: 1.0000
