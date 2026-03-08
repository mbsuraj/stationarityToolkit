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

def run_all_seasonal_tests(ts: pd.Series, alpha: float = 0.05) -> list:
    """Run all seasonal tests and return results."""
    return [
        acf_peak_test(ts, alpha),
        stl_test(ts, alpha)
    ]
