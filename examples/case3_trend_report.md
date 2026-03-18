# Stationarity Detection Report

## Summary

- Trend Stationary: ❌ No
- Variance Stationary: ❌ No
- Seasonal Stationary: ❌ No

## Trend Tests

### Augmented Dickey-Fuller (ADF) Test

- Result: ❌ Non-stationary
- Note: Deterministic trend detected - stationary after detrending
- Interpretation: H0: Unit root. ADF-c p=0.9116 >= 0.05, ADF-ct p=0.0000 < 0.05. Deterministic trend.
- Statistic: -0.3909
- P-value: 0.9116

### Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test

- Result: ❌ Non-stationary
- Note: Deterministic trend detected - stationary after detrending
- Interpretation: H0: Stationary. KPSS-c p=0.0100 <= 0.05, KPSS-ct p=0.1000 > 0.05. Deterministic trend.
- Statistic: 5.1008
- P-value: 0.0100

### Phillips-Perron (PP) Test

- Result: ❌ Non-stationary
- Note: Deterministic trend detected - stationary after detrending
- Interpretation: H0: Unit root. PP-c p=0.8732 >= 0.05, PP-ct p=0.0000 < 0.05. Deterministic trend.
- Statistic: -0.5899
- P-value: 0.8732

### Zivot-Andrews Test for Structural Breaks

- Result: ✅ Stationary
- Note: Stationary with structural breaks: level shift at obs 743, trend shift at obs 305, level+trend shift at obs 743. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise.
- Interpretation: H0: Unit root with no break. ZA-c p=0.0000, ZA-t p=0.0007, ZA-ct p=0.0002 < 0.05. Reject H0 - stationary.
- Statistic: -32.1303
- P-value: 0.0000

## Variance Tests

### Levene's Test for Variance Homogeneity

- Result: ✅ Stationary
- Note: Constant variance across time
- Interpretation: H0: Equal variances across segments. Levene p=0.9614 > 0.05. Fail to reject H0.
- Statistic: 0.0976
- P-value: 0.9614

### Bartlett's Test for Variance Homogeneity

- Result: ✅ Stationary
- Note: Constant variance across time
- Interpretation: H0: Equal variances across segments (assumes normality). Bartlett p=0.9890 > 0.05. Fail to reject H0.
- Statistic: 0.1229
- P-value: 0.9890

### White's Test for Heteroskedasticity

- Result: ✅ Stationary
- Note: Constant variance across time
- Interpretation: H0: Homoskedasticity. White p=0.6645 > 0.05. Fail to reject H0.
- Statistic: 0.8175
- P-value: 0.6645

### ARCH Test for Conditional Heteroskedasticity

- Result: ❌ Non-stationary
- Note: Volatility clustering detected - consider GARCH modeling (may trigger on trend/seasonality if present)
- Interpretation: H0: No ARCH effects. ARCH p=0.0000 <= 0.05. Reject H0.
- Statistic: 984.8041
- P-value: 0.0000

## Seasonal Tests

### ACF/PACF Peak Detection

- Result: ❌ Non-stationary
- Note: Seasonality detected (periods: 7, 30) - consider seasonal differencing (may trigger on trend/variance)
- Interpretation: H0: No seasonality. Ljung-Box p=0.0000 < 0.05. Reject H0.
- Statistic: 0.9923
- P-value: 0.0000

### STL Decomposition

- Result: ✅ Stationary
- Note: No significant seasonal component detected
- Interpretation: H0: No seasonality. F-stat p=1.0000 > 0.05. Fail to reject H0.
- Statistic: 0.1486
- P-value: 1.0000
