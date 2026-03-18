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
- Statistic: 86.2640
- P-value: 0.0000

### Bartlett's Test for Variance Homogeneity

- Result: ❌ Non-stationary
- Note: Variance changes detected - consider Box-Cox or Yeo-Johnson transform
- Interpretation: H0: Equal variances across segments (assumes normality). Bartlett p=0.0000 <= 0.05. Reject H0.
- Statistic: 815.8958
- P-value: 0.0000

### White's Test for Heteroskedasticity

- Result: ❌ Non-stationary
- Note: Time-dependent variance detected - consider Box-Cox or Yeo-Johnson transform
- Interpretation: H0: Homoskedasticity. White p=0.0000 <= 0.05. Reject H0.
- Statistic: 52.7225
- P-value: 0.0000

### ARCH Test for Conditional Heteroskedasticity

- Result: ❌ Non-stationary
- Note: Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)
- Interpretation: H0: No ARCH effects. ARCH p=0.0012 <= 0.05. Reject H0.
- Statistic: 29.1246
- P-value: 0.0012

## Seasonal Tests

### ACF/PACF Peak Detection

- Result: ❌ Non-stationary
- Note: Seasonality detected (periods: 7, 30) - consider seasonal differencing (may trigger on trend/variance)
- Interpretation: H0: No seasonality. Ljung-Box p=0.0000 < 0.05. Reject H0.
- Statistic: 0.1372
- P-value: 0.0000

### STL Decomposition

- Result: ✅ Stationary
- Note: No significant seasonal component detected
- Interpretation: H0: No seasonality. F-stat p=1.0000 > 0.05. Fail to reject H0.
- Statistic: 0.2474
- P-value: 1.0000
