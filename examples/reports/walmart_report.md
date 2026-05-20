# Stationarity Detection Report

## Summary

- Trend Stationary: ❌ No
- Variance Stationary: ❌ No
- Seasonal Stationary: ❌ No

## Trend Tests

### Augmented Dickey-Fuller (ADF) Test

- Result: ✅ Stationary
- Note: Stationary around constant mean
- Interpretation: H0: Unit root. ADF-c p=0.0155 < 0.05, ADF-ct p=0.0039 < 0.05. Reject H0.
- Statistic: -3.2863
- P-value: 0.0155

### Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test

- Result: ❌ Non-stationary
- Note: Deterministic trend detected - stationary after detrending
- Interpretation: H0: Stationary. KPSS-c p=0.0100 <= 0.05, KPSS-ct p=0.0951 > 0.05. Deterministic trend.
- Statistic: 1.1964
- P-value: 0.0100

### Phillips-Perron (PP) Test

- Result: ✅ Stationary
- Note: Stationary around constant mean
- Interpretation: H0: Unit root. PP-c p=0.0000 < 0.05, PP-ct p=0.0000 < 0.05. Reject H0.
- Statistic: -7.8514
- P-value: 0.0000

### Zivot-Andrews Test for Structural Breaks

- Result: ✅ Stationary
- Note: Stationary with structural breaks: level shift at obs 99, trend shift at obs 155, level+trend shift at obs 99. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.
- Interpretation: H0: Unit root with no break. ZA-c p=0.0040, ZA-t p=0.0285, ZA-ct p=0.0157 < 0.05. Reject H0 - stationary.
- Statistic: -5.5103
- P-value: 0.0040

## Variance Tests

### Levene's Test for Variance Homogeneity

- Result: ❌ Non-stationary
- Note: Variance changes detected - consider Box-Cox or Yeo-Johnson transform
- Interpretation: H0: Equal variances across segments. Levene p=0.0008 <= 0.05. Reject H0.
- Statistic: 5.7629
- P-value: 0.0008

### Bartlett's Test for Variance Homogeneity

- Result: ❌ Non-stationary
- Note: Variance changes detected - consider Box-Cox or Yeo-Johnson transform
- Interpretation: H0: Equal variances across segments (assumes normality). Bartlett p=0.0000 <= 0.05. Reject H0.
- Statistic: 30.8232
- P-value: 0.0000

### White's Test for Heteroskedasticity

- Result: ❌ Non-stationary
- Note: Time-dependent variance detected - consider Box-Cox or Yeo-Johnson transform
- Interpretation: H0: Homoskedasticity. White p=0.0045 <= 0.05. Reject H0.
- Statistic: 10.8138
- P-value: 0.0045

### ARCH Test for Conditional Heteroskedasticity

- Result: ❌ Non-stationary
- Note: Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)
- Interpretation: H0: No ARCH effects. ARCH p=0.0000 <= 0.05. Reject H0.
- Statistic: 173.6824
- P-value: 0.0000

## Seasonal Tests

### ACF/PACF Peak Detection

- Result: ❌ Non-stationary
- Note: Seasonality detected (periods: 52) - consider seasonal differencing (may trigger on trend/variance)
- Interpretation: H0: No seasonality. Ljung-Box p=0.0000 < 0.05. Reject H0.
- Statistic: 0.7508
- P-value: 0.0000

### STL Decomposition

- Result: ❌ Non-stationary
- Note: Significant seasonal component detected (period 52) - consider seasonal differencing
- Interpretation: H0: No seasonality. F-stat p=0.0000 <= 0.05. Reject H0.
- Statistic: 2.3183
- P-value: 0.0000
