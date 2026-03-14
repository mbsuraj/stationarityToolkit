# Stationarity Detection Report

## Summary

- Trend Stationary: ✅ Yes
- Variance Stationary: ❌ No
- Seasonal Stationary: ✅ Yes

## Trend Tests

All tests passed ✅

## Variance Tests

### Levene's Test for Variance Homogeneity

- Result: ❌ Non-stationary
- Note: Variance changes detected - consider Box-Cox or Yeo-Johnson transform
- Interpretation: H0: Equal variances across segments. Levene p=0.0000 <= 0.05. Reject H0.
- Statistic: 29.9321
- P-value: 0.0000

### Bartlett's Test for Variance Homogeneity

- Result: ❌ Non-stationary
- Note: Variance changes detected - consider Box-Cox or Yeo-Johnson transform
- Interpretation: H0: Equal variances across segments (assumes normality). Bartlett p=0.0000 <= 0.05. Reject H0.
- Statistic: 119.2368
- P-value: 0.0000

### White's Test for Heteroskedasticity

- Result: ❌ Non-stationary
- Note: Time-dependent variance detected - consider Box-Cox or Yeo-Johnson transform
- Interpretation: H0: Homoskedasticity. White p=0.0000 <= 0.05. Reject H0.
- Statistic: 72.7950
- P-value: 0.0000

### ARCH Test for Conditional Heteroskedasticity

- Result: ✅ Stationary
- Note: No volatility clustering detected (ARCH tests for clustered variance, not smooth changes)
- Interpretation: H0: No ARCH effects. ARCH p=0.6826 > 0.05. Fail to reject H0.
- Statistic: 7.4477
- P-value: 0.6826

## Seasonal Tests

All tests passed ✅
