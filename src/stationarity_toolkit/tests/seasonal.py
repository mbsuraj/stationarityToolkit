"""Seasonal stationarity tests."""

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.signal import periodogram
from scipy.stats import f as f_dist
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
import warnings

from ..results import TestResult
from ..utils import get_contextual_periods


def acf_peak_test(
    timeseries: pd.Series,
    alpha: float = 0.05,
    nlags: int = None,
    threshold: float = 2.0
) -> TestResult:
    """
    Detect seasonality using ACF/PACF peak detection at regular lags.
    
    This test looks for significant peaks in the autocorrelation function (ACF)
    at regular intervals, which indicate periodic patterns (seasonality).
    
    Args:
        timeseries: Input time series
        alpha: Significance level
        nlags: Number of lags to compute (default: min(len(ts)//2, 40))
        threshold: Number of standard errors for significance (default: 2.0)
        
    Returns:
        TestResult object with peak detection results
    """
   
    ts = timeseries.dropna()
    
    if len(ts) < 20:
        raise ValueError(
            f"Time series too short for ACF peak test. "
            f"Need at least 20 observations, got {len(ts)}"
        )
    
    expected_periods = get_contextual_periods(ts)
    max_expected = max(expected_periods) if expected_periods else 40
    
    if nlags is None:
        # Need at least 2 multiples of largest expected period
        min_required = max_expected * 2 if expected_periods else 40
        nlags = min(len(ts) // 2, max(min_required, 40))
    
    try:
        # Use Ljung-Box test on seasonal lags for statistical rigor
        has_seasonality = False
        detected_periods = []
        min_p_value = 1.0
        
        for period in expected_periods:
            if period >= len(ts) // 3:
                continue
            
            # Test multiple consecutive multiples of the period
            max_mult = min(3, nlags // period)
            if max_mult < 2:
                continue
                
            seasonal_lags = [period * i for i in range(1, max_mult + 1)]
            
            try:
                lb_result = acorr_ljungbox(ts, lags=seasonal_lags, return_df=True)
                # Check if at least 2 consecutive multiples are significant
                significant_lags = (lb_result['lb_pvalue'] < alpha).sum()
                
                if significant_lags >= 2:
                    has_seasonality = True
                    detected_periods.append(period)
                    min_p_value = min(min_p_value, lb_result['lb_pvalue'].min())
            except:
                continue
        
        is_stationary = not has_seasonality
        
        # Calculate test statistic as max absolute ACF value for reporting
        acf_values = acf(ts.values, nlags=min(nlags, len(ts) - 1), fft=False)
        test_statistic = max(abs(acf_values[1:]))
        p_value = min_p_value
        
        if has_seasonality:
            periods_str = ", ".join(map(str, detected_periods))
            educational_note = f"Seasonality detected (periods: {periods_str}) - consider seasonal differencing (may trigger on trend/variance)"
            interpretation = f"H0: No seasonality. Ljung-Box p={p_value:.4f} < {alpha}. Reject H0."
        else:
            educational_note = "No seasonal patterns detected"
            interpretation = f"H0: No seasonality. Ljung-Box p={p_value:.4f} >= {alpha}. Fail to reject H0."
        
        return TestResult(
            test_name="ACF/PACF Peak Detection",
            statistic=test_statistic,
            p_value=p_value,
            is_stationary=is_stationary,
            interpretation=interpretation,
            educational_note=educational_note
        )
    
    except Exception as e:
        warnings.warn(f"ACF peak test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="ACF/PACF Peak Detection",
            statistic=np.nan,
            p_value=np.nan,
            is_stationary=False,
            interpretation=f"Test failed: {str(e)}",
            educational_note="Test execution failed."
        )


def stl_test(
    timeseries: pd.Series,
    alpha: float = 0.05,
    period: int = None
) -> TestResult:
    """
    Test for seasonality using STL decomposition with significance testing.
    
    Decomposes the series into trend, seasonal, and residual components using STL.
    Tests if the seasonal component is significant by comparing its variance
    to the residual variance.
    
    Args:
        timeseries: Input time series
        alpha: Significance level
        period: Seasonal period (auto-detected if None)
        
    Returns:
        TestResult object with STL decomposition results
    """
    ts = timeseries.dropna()
    
    if len(ts) < 20:
        raise ValueError(
            f"Time series too short for STL test. "
            f"Need at least 20 observations, got {len(ts)}"
        )
    
    try:
        if period is None:
            expected_periods = get_contextual_periods(ts)
            period = expected_periods[0] if expected_periods else 7
        
        if period < 2 or period >= len(ts) // 2:
            period = min(7, len(ts) // 3)
        
        # Run STL decomposition
        stl = STL(ts, period=period, seasonal=13)
        result = stl.fit()
        
        # Test significance: compare seasonal variance to residual variance
        seasonal_var = np.var(result.seasonal)
        residual_var = np.var(result.resid)
        
        # F-statistic: ratio of variances
        if residual_var > 0:
            f_stat = seasonal_var / residual_var
        else:
            f_stat = np.inf
        
        # Degrees of freedom
        df1 = len(ts) - 1
        df2 = len(ts) - 1
        
        # P-value from F-distribution
        p_value = 1 - f_dist.cdf(f_stat, df1, df2)
        
        is_stationary = p_value > alpha
        
        if not is_stationary:
            educational_note = f"Significant seasonal component detected (period {period}) - consider seasonal differencing"
            interpretation = f"H0: No seasonality. F-stat p={p_value:.4f} <= {alpha}. Reject H0."
        else:
            educational_note = "No significant seasonal component detected"
            interpretation = f"H0: No seasonality. F-stat p={p_value:.4f} > {alpha}. Fail to reject H0."
        
        return TestResult(
            test_name="STL Decomposition",
            statistic=f_stat,
            p_value=p_value,
            is_stationary=is_stationary,
            interpretation=interpretation,
            educational_note=educational_note
        )
    
    except Exception as e:
        warnings.warn(f"STL test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="STL Decomposition",
            statistic=np.nan,
            p_value=np.nan,
            is_stationary=False,
            interpretation=f"Test failed: {str(e)}",
            educational_note="Test execution failed."
        )


def ocsb_test(
    timeseries: pd.Series,
    alpha: float = 0.05,
    period: int = None
) -> TestResult:
    """
    OCSB (Osborn-Chui-Smith-Birchenhall) test for seasonal unit roots.
    
    Tests for unit roots at seasonal frequencies. The null hypothesis is that
    seasonal unit roots are present (non-stationary). Rejection indicates
    seasonal stationarity.
    
    Args:
        timeseries: Input time series
        alpha: Significance level
        period: Seasonal period (auto-detected if None)
        
    Returns:
        TestResult object with OCSB test results
    """
    ts = timeseries.dropna()
    
    if len(ts) < 20:
        raise ValueError(
            f"Time series too short for OCSB test. "
            f"Need at least 20 observations, got {len(ts)}"
        )
    
    try:
        if period is None:
            expected_periods = get_contextual_periods(ts)
            period = expected_periods[0] if expected_periods else 7
        
        if period < 2 or period >= len(ts) // 2:
            period = min(7, len(ts) // 3)
        
        y = ts.values
        n = len(y)
        
        # OCSB test: regress seasonal differences on lagged values
        # For period S, test: Δ_S y_t = α + β*y_{t-S} + Σ(γ_i * Δ_S y_{t-i}) + ε
        
        S = period
        
        # Compute seasonal difference: Δ_S y_t = y_t - y_{t-S}
        if n <= S:
            raise ValueError(f"Time series too short for period {S}")
        
        y_diff = y[S:] - y[:-S]
        y_lag = y[:-S]
        
        # Create lagged differences for augmentation
        max_lags = min(4, (len(y_diff) - 1) // 2)
        
        X = [np.ones(len(y_diff) - max_lags), y_lag[max_lags:]]
        
        for i in range(1, max_lags + 1):
            if len(y_diff) > i:
                X.append(y_diff[max_lags - i:-i] if i < max_lags else y_diff[:-i])
        
        X = np.column_stack(X)
        y_reg = y_diff[max_lags:]
        
        # OLS estimation
        beta, residuals_sum, _, _ = lstsq(X, y_reg)
        
        # Test statistic: t-statistic on coefficient of y_{t-S}
        residuals = y_reg - X @ beta
        sigma2 = np.var(residuals, ddof=X.shape[1])
        
        # Standard error of beta[1] (coefficient on y_{t-S})
        XtX_inv = np.linalg.inv(X.T @ X + 1e-10 * np.eye(X.shape[1]))
        se_beta = np.sqrt(sigma2 * XtX_inv[1, 1])
        
        # t-statistic
        t_stat = beta[1] / (se_beta + 1e-10)
        
        # Critical values (approximate, similar to ADF)
        # More negative = reject null of unit root
        critical_values = {
            0.01: -3.43,
            0.05: -2.86,
            0.10: -2.57
        }
        
        critical_value = critical_values.get(alpha, -2.86)
        
        # Null hypothesis: seasonal unit root present (non-stationary)
        # Reject if t_stat < critical_value (more negative)
        is_stationary = t_stat < critical_value
        
        # Approximate p-value
        if t_stat < critical_values[0.01]:
            p_value = 0.005
        elif t_stat < critical_values[0.05]:
            p_value = 0.025
        elif t_stat < critical_values[0.10]:
            p_value = 0.075
        else:
            p_value = 0.15
        
        if is_stationary:
            educational_note = f"No seasonal unit roots detected (period {period}). OCSB detects random-walk seasonality, not fixed patterns - use STL for deterministic seasonality"
            interpretation = f"H0: Seasonal unit root present. OCSB t-stat={t_stat:.4f} < {critical_value:.4f}. Reject H0."
        else:
            educational_note = f"Seasonal unit roots detected (period {period}) - consider seasonal differencing (may trigger on trend)"
            interpretation = f"H0: Seasonal unit root present. OCSB t-stat={t_stat:.4f} >= {critical_value:.4f}. Fail to reject H0."
        
        return TestResult(
            test_name="OCSB",
            statistic=t_stat,
            p_value=p_value,
            is_stationary=is_stationary,
            interpretation=interpretation,
            educational_note=educational_note
        )
    
    except Exception as e:
        warnings.warn(f"OCSB test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="OCSB",
            statistic=np.nan,
            p_value=np.nan,
            is_stationary=False,
            interpretation=f"Test failed: {str(e)}",
            educational_note="Test execution failed."
        )


def canova_hansen_test(
    timeseries: pd.Series,
    alpha: float = 0.05,
    period: int = None
) -> TestResult:
    """
    Canova-Hansen test for seasonal unit roots.
    
    Tests the null hypothesis that seasonal unit roots are present (non-stationary).
    Rejection of the null indicates seasonal stationarity.
    
    Args:
        timeseries: Input time series
        alpha: Significance level
        period: Seasonal period (auto-detected if None)
        
    Returns:
        TestResult object with Canova-Hansen test results
    """
    ts = timeseries.dropna()
    
    if len(ts) < 20:
        raise ValueError(
            f"Time series too short for Canova-Hansen test. "
            f"Need at least 20 observations, got {len(ts)}"
        )
    
    try:
        if period is None:
            expected_periods = get_contextual_periods(ts)
            period = expected_periods[0] if expected_periods else 7
        
        if period < 2 or period >= len(ts) // 2:
            period = min(7, len(ts) // 3)
        
        y = ts.values
        n = len(y)
        
        # Create seasonal dummy variables
        S = period
        dummies = np.zeros((n, S))
        for i in range(n):
            dummies[i, i % S] = 1
        
        # Create time trend
        t = np.arange(1, n + 1).reshape(-1, 1)
        
        # Regression: y_t = α + βt + Σ(γ_s * D_s) + ε
        X = np.column_stack([np.ones(n), t, dummies])
        
        # OLS estimation
        beta, _, _, _ = lstsq(X, y)
        
        # Residuals
        residuals = y - X @ beta
        
        # Compute test statistic
        # CH statistic is based on seasonal dummy coefficients
        gamma = beta[2:]  # Seasonal coefficients
        
        # Variance of residuals
        sigma2 = np.var(residuals, ddof=X.shape[1])
        
        # Compute Wald-type statistic
        # Test if seasonal dummies are jointly zero
        test_stat = np.sum(gamma**2) / (sigma2 + 1e-10)
        
        # Critical values (approximate, from Canova-Hansen 1995)
        # These are rough approximations for different significance levels
        critical_values = {
            0.01: 1.0,
            0.05: 0.75,
            0.10: 0.65
        }
        
        critical_value = critical_values.get(alpha, 0.75)
        
        # Null hypothesis: seasonal unit roots present (non-stationary)
        # Reject null if test_stat > critical_value
        is_stationary = test_stat > critical_value
        
        # Approximate p-value
        if test_stat > critical_values[0.01]:
            p_value = 0.005
        elif test_stat > critical_values[0.05]:
            p_value = 0.025
        elif test_stat > critical_values[0.10]:
            p_value = 0.075
        else:
            p_value = 0.15
        
        if is_stationary:
            educational_note = f"No seasonal unit roots detected (period {period})"
            interpretation = f"H0: Seasonal unit roots present. CH p={p_value:.4f} <= {alpha}. Reject H0."
        else:
            educational_note = f"Seasonal unit roots detected (period {period}) - consider seasonal differencing"
            interpretation = f"H0: Seasonal unit roots present. CH p={p_value:.4f} > {alpha}. Fail to reject H0."
        
        return TestResult(
            test_name="Canova-Hansen",
            statistic=test_stat,
            p_value=p_value,
            is_stationary=is_stationary,
            interpretation=interpretation,
            educational_note=educational_note
        )
    
    except Exception as e:
        warnings.warn(f"Canova-Hansen test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="Canova-Hansen",
            statistic=np.nan,
            p_value=np.nan,
            is_stationary=False,
            interpretation=f"Test failed: {str(e)}",
            educational_note="Test execution failed."
        )


def spectral_test(
    timeseries: pd.Series,
    alpha: float = 0.05,
    threshold: float = 3.0
) -> TestResult:
    """
    Detect seasonality using spectral analysis (periodogram peaks).
    
    Computes the periodogram and identifies significant peaks that indicate
    periodic patterns in the data.
    
    Args:
        timeseries: Input time series
        alpha: Significance level
        threshold: Threshold for peak detection (multiples of median power)
        
    Returns:
        TestResult object with spectral analysis results
    """
    ts_series = timeseries.dropna()
    expected_periods = get_contextual_periods(ts_series)
    ts = ts_series.values
    
    if len(ts) < 20:
        raise ValueError(
            f"Time series too short for spectral test. "
            f"Need at least 20 observations, got {len(ts)}"
        )
    
    try:
        # Compute periodogram
        freqs, power = periodogram(ts)
        
        # Remove DC component (frequency 0)
        freqs = freqs[1:]
        power = power[1:]
        
        # Find peaks above threshold
        median_power = np.median(power)
        threshold_power = threshold * median_power
        
        significant_peaks = []
        for i in range(1, len(power) - 1):
            if power[i] > threshold_power and power[i] > power[i-1] and power[i] > power[i+1]:
                period = 1 / freqs[i] if freqs[i] > 0 else np.inf
                significant_peaks.append((period, power[i]))
        
        # ONLY flag seasonality if peaks match expected periods with tolerance
        has_seasonality = False
        matched_periods = []
        
        for period, pwr in significant_peaks:
            for exp_period in expected_periods:
                # Adaptive tolerance: tighter for short periods
                tolerance = 0.1 if exp_period < 20 else 0.15
                if abs(period - exp_period) / exp_period < tolerance:
                    has_seasonality = True
                    matched_periods.append((period, exp_period))
                    break
        
        is_stationary = not has_seasonality
        
        # Fisher's g-test for statistical rigor
        if len(power) > 0:
            g_stat = np.max(power) / np.sum(power)
            # Approximate p-value for Fisher's g-test
            n_freqs = len(power)
            p_value = 1.0 - (1.0 - g_stat) ** n_freqs
            test_statistic = g_stat
        else:
            test_statistic = 0
            p_value = 1.0
        
        if has_seasonality:
            # Show matched periods with their detected values
            periods_str = ", ".join([f"{p:.1f}≈{int(ep)}" for p, ep in matched_periods[:3]])
            educational_note = f"Periodic components detected (periods: {periods_str}) - consider seasonal differencing"
            interpretation = f"H0: No periodicity. Spectral peaks match expected periods (Fisher's g={test_statistic:.4f}). Reject H0."
        else:
            educational_note = "No significant periodic components detected"
            interpretation = f"H0: No periodicity. No peaks match expected periods (Fisher's g={test_statistic:.4f}). Fail to reject H0."
        
        return TestResult(
            test_name="Spectral Analysis",
            statistic=test_statistic,
            p_value=p_value,
            is_stationary=is_stationary,
            interpretation=interpretation,
            educational_note=educational_note
        )
    
    except Exception as e:
        warnings.warn(f"Spectral test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="Spectral Analysis",
            statistic=np.nan,
            p_value=np.nan,
            is_stationary=False,
            interpretation=f"Test failed: {str(e)}",
            educational_note="Test execution failed."
        )


def contextual_period_test(
    timeseries: pd.Series,
    alpha: float = 0.05
) -> TestResult:
    """
    Test for seasonality at contextual periods based on data frequency.
    
    Args:
        timeseries: Input time series with datetime index
        alpha: Significance level
        
    Returns:
        TestResult object with contextual period detection results
    """
    ts = timeseries.dropna()
    
    if len(ts) < 20:
        raise ValueError(f"Time series too short. Need at least 20 observations, got {len(ts)}")
    
    try:
        periods = get_contextual_periods(ts)
        
        if not periods:
            return TestResult(
                test_name="Contextual Period Detection",
                statistic=0.0,
                p_value=1.0,
                is_stationary=True,
                interpretation="No contextual periods to test for this frequency.",
                educational_note="Contextual period detection checks for seasonality at expected periods based on data frequency."
            )
        
        # Test ACF at contextual periods
        max_lag = max(periods)
        if max_lag >= len(ts):
            max_lag = len(ts) - 1
            periods = [p for p in periods if p < len(ts)]
        
        if not periods:
            return TestResult(
                test_name="Contextual Period Detection",
                statistic=0.0,
                p_value=1.0,
                is_stationary=True,
                interpretation="Time series too short for contextual periods.",
                educational_note="Contextual period detection checks for seasonality at expected periods based on data frequency."
            )
        
        acf_values = acf(ts.values, nlags=max_lag, fft=False)
        
        # Check ACF at contextual periods
        conf_interval = 2.0 / np.sqrt(len(ts))
        significant_periods = []
        
        for period in periods:
            if period < len(acf_values) and abs(acf_values[period]) > conf_interval:
                significant_periods.append((period, acf_values[period]))
        
        has_seasonality = len(significant_periods) > 0
        is_stationary = not has_seasonality
        
        test_statistic = max([abs(acf_values[p]) for p in periods if p < len(acf_values)], default=0)
        p_value = 1.0 - (test_statistic / conf_interval) if test_statistic > conf_interval else 1.0
        p_value = max(0.0, min(1.0, p_value))
        
        if has_seasonality:
            periods_str = ", ".join([f"{p}" for p, _ in significant_periods])
            interpretation = (
                f"Contextual period detection found significant ACF at periods: {periods_str}. "
                f"Max ACF: {test_statistic:.4f} (threshold: {conf_interval:.4f}). "
                f"The series appears to have seasonal patterns at expected frequencies."
            )
        else:
            interpretation = (
                f"Contextual period detection found no significant ACF at expected periods {periods}. "
                f"Max ACF: {test_statistic:.4f} (threshold: {conf_interval:.4f}). "
                f"The series appears to be seasonally stationary."
            )
        
        educational_note = (
            "Contextual period detection checks for seasonality at expected periods based on data frequency "
            "(e.g., 7/30/365 for daily data, 12 for monthly data)."
        )
        
        return TestResult(
            test_name="Contextual Period Detection",
            statistic=test_statistic,
            p_value=p_value,
            is_stationary=is_stationary,
            interpretation=interpretation,
            educational_note=educational_note
        )
    
    except Exception as e:
        warnings.warn(f"Contextual period test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="Contextual Period Detection",
            statistic=np.nan,
            p_value=np.nan,
            is_stationary=False,
            interpretation=f"Test failed: {str(e)}",
            educational_note="Test execution failed."
        )


def contextual_rolling_test(
    timeseries: pd.Series,
    alpha: float = 0.05
) -> TestResult:
    """
    Test for seasonality using rolling statistics at contextual periods.
    
    Args:
        timeseries: Input time series with datetime index
        alpha: Significance level
        
    Returns:
        TestResult object with contextual rolling statistics results
    """
    ts = timeseries.dropna()
    
    if len(ts) < 20:
        raise ValueError(f"Time series too short. Need at least 20 observations, got {len(ts)}")
    
    try:
        periods = get_contextual_periods(ts)
        
        if not periods:
            return TestResult(
                test_name="Contextual Rolling Statistics",
                statistic=0.0,
                p_value=1.0,
                is_stationary=True,
                interpretation="No contextual periods to test.",
                educational_note="Contextual rolling statistics test checks if rolling means vary systematically at expected seasonal periods."
            )
        
        # Test rolling mean variance at contextual periods
        max_variance = 0
        detected_period = None
        
        for period in periods:
            if period >= len(ts) // 2:
                continue
            
            rolling_mean = ts.rolling(window=period).mean().dropna()
            
            if len(rolling_mean) < 2:
                continue
            
            variance = np.var(rolling_mean)
            
            if variance > max_variance:
                max_variance = variance
                detected_period = period
        
        # Compare to overall variance
        overall_variance = np.var(ts)
        
        if overall_variance > 0:
            variance_ratio = max_variance / overall_variance
        else:
            variance_ratio = 0
        
        # Threshold for significance
        threshold = 0.1
        has_seasonality = variance_ratio > threshold
        is_stationary = not has_seasonality
        
        test_statistic = variance_ratio
        p_value = 1.0 - variance_ratio if variance_ratio < 1.0 else 0.01
        
        if has_seasonality:
            interpretation = (
                f"Contextual rolling statistics detected systematic variation at period {detected_period}. "
                f"Variance ratio: {variance_ratio:.4f} (threshold: {threshold:.4f}). "
                f"The series appears to have seasonal patterns."
            )
        else:
            interpretation = (
                f"Contextual rolling statistics found no systematic variation at expected periods. "
                f"Variance ratio: {variance_ratio:.4f} (threshold: {threshold:.4f}). "
                f"The series appears to be seasonally stationary."
            )
        
        educational_note = (
            "Contextual rolling statistics test checks if rolling means vary systematically "
            "at expected seasonal periods, indicating seasonal patterns."
        )
        
        return TestResult(
            test_name="Contextual Rolling Statistics",
            statistic=test_statistic,
            p_value=p_value,
            is_stationary=is_stationary,
            interpretation=interpretation,
            educational_note=educational_note
        )
    
    except Exception as e:
        warnings.warn(f"Contextual rolling test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="Contextual Rolling Statistics",
            statistic=np.nan,
            p_value=np.nan,
            is_stationary=False,
            interpretation=f"Test failed: {str(e)}",
            educational_note="Test execution failed."
        )


def run_all_seasonal_tests(ts: pd.Series, alpha: float = 0.05) -> list:
    """Run all seasonal tests and return results."""
    return [
        acf_peak_test(ts, alpha),
        stl_test(ts, alpha),
        ocsb_test(ts, alpha)
    ]
