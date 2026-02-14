"""
Variance stationarity tests.

This module provides proper tests for detecting heteroskedasticity (non-constant variance)
in time series data. These are the CORRECT tests for variance stationarity, unlike
Phillips-Perron which tests for unit roots (trend stationarity).
"""

from typing import Tuple, Dict
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_arch
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings

from ..results import TestResult


def levene_test(
    timeseries: pd.Series,
    alpha: float = 0.05,
    window_size: int = None
) -> TestResult:
    """
    Perform Levene's test for homogeneity of variance.
    
    Levene's test is robust to departures from normality and tests whether
    the variance is constant across different segments of the time series.
    
    Null Hypothesis: All segments have equal variances (variance is stationary)
    Alternative: At least one segment has different variance
    
    Args:
        timeseries: Input time series
        alpha: Significance level
        window_size: Size of segments to compare (default: len(ts)//4)
        
    Returns:
        TestResult object with test statistics and interpretation
    """
    ts = timeseries.dropna().values
    n = len(ts)
    
    if window_size is None:
        window_size = int(max(n // 4, 10))
    
    # Split into segments
    n_segments = int(n // window_size)
    if n_segments < 2:
        raise ValueError(
            f"Time series too short for Levene test. "
            f"Need at least {2 * window_size} observations, got {n}"
        )
    
    segments = [
        ts[i*window_size:(i+1)*window_size] 
        for i in range(n_segments)
    ]
    
    # Perform Levene's test
    statistic, p_value = stats.levene(*segments, center='median')
    
    is_stationary = p_value > alpha
    
    interpretation = (
        f"Levene's test divides the series into {int(n_segments)} segments of size {int(window_size)}. "
        f"With p-value = {p_value:.4f} {'>' if is_stationary else '<='} α = {alpha}, "
        f"we {'fail to reject' if is_stationary else 'reject'} the null hypothesis. "
        f"The variance appears to be {'constant' if is_stationary else 'non-constant'} "
        f"across the time series."
    )
    
    return TestResult(
        test_name="Levene's Test for Variance Homogeneity",
        test_statistic=statistic,
        p_value=p_value,
        critical_values={},  # Levene's test doesn't use critical values
        is_stationary=is_stationary,
        alpha=alpha,
        interpretation=interpretation
    )


def bartlett_test(
    timeseries: pd.Series,
    alpha: float = 0.05,
    window_size: int = None
) -> TestResult:
    """
    Perform Bartlett's test for homogeneity of variance.
    
    Bartlett's test is more sensitive to departures from normality than Levene's test
    but can be more powerful when data is normally distributed.
    
    Null Hypothesis: All segments have equal variances (variance is stationary)
    Alternative: At least one segment has different variance
    
    Args:
        timeseries: Input time series
        alpha: Significance level
        window_size: Size of segments to compare (default: len(ts)//4)
        
    Returns:
        TestResult object with test statistics and interpretation
    """
    ts = timeseries.dropna().values
    n = len(ts)
    
    if window_size is None:
        window_size = int(max(n // 4, 10))
    
    # Split into segments
    n_segments = int(n // window_size)
    if n_segments < 2:
        raise ValueError(
            f"Time series too short for Bartlett test. "
            f"Need at least {2 * window_size} observations, got {n}"
        )
    
    segments = [
        ts[i*window_size:(i+1)*window_size] 
        for i in range(n_segments)
    ]
    
    # Perform Bartlett's test
    statistic, p_value = stats.bartlett(*segments)
    
    is_stationary = p_value > alpha
    
    interpretation = (
        f"Bartlett's test divides the series into {int(n_segments)} segments of size {int(window_size)}. "
        f"With p-value = {p_value:.4f} {'>' if is_stationary else '<='} α = {alpha}, "
        f"we {'fail to reject' if is_stationary else 'reject'} the null hypothesis. "
        f"The variance appears to be {'constant' if is_stationary else 'non-constant'} "
        f"across the time series. Note: This test assumes normality."
    )
    
    return TestResult(
        test_name="Bartlett's Test for Variance Homogeneity",
        test_statistic=statistic,
        p_value=p_value,
        critical_values={},
        is_stationary=is_stationary,
        alpha=alpha,
        interpretation=interpretation
    )


def arch_test(
    timeseries: pd.Series,
    alpha: float = 0.05,
    lags: int = None
) -> TestResult:
    """
    Perform ARCH (Autoregressive Conditional Heteroskedasticity) test.
    
    The ARCH test specifically tests for time-varying volatility, which is
    common in financial time series. It tests whether the variance depends
    on past squared residuals.
    
    Null Hypothesis: No ARCH effects (constant variance)
    Alternative: ARCH effects present (time-varying variance)
    
    Args:
        timeseries: Input time series
        alpha: Significance level
        lags: Number of lags to test (default: min(10, len(ts)//5))
        
    Returns:
        TestResult object with test statistics and interpretation
    """
    ts = timeseries.dropna().values
    n = len(ts)
    
    if lags is None:
        lags = min(10, n // 5)
    
    if n < lags + 10:
        raise ValueError(
            f"Time series too short for ARCH test with {lags} lags. "
            f"Need at least {lags + 10} observations, got {n}"
        )
    
    try:
        # Perform ARCH test
        lm_statistic, lm_pvalue, f_statistic, f_pvalue = het_arch(ts, nlags=lags)
        
        is_stationary = lm_pvalue > alpha
        
        interpretation = (
            f"ARCH test with {lags} lags tests for autoregressive conditional heteroskedasticity. "
            f"LM statistic = {lm_statistic:.4f}, p-value = {lm_pvalue:.4f}. "
            f"With p-value {'>' if is_stationary else '<='} α = {alpha}, "
            f"we {'fail to reject' if is_stationary else 'reject'} the null hypothesis. "
            f"{'No significant' if is_stationary else 'Significant'} ARCH effects detected, "
            f"suggesting {'constant' if is_stationary else 'time-varying'} variance."
        )
        
        return TestResult(
            test_name="ARCH Test for Conditional Heteroskedasticity",
            test_statistic=lm_statistic,
            p_value=lm_pvalue,
            critical_values={"F-statistic": f_statistic, "F-pvalue": f_pvalue},
            is_stationary=is_stationary,
            alpha=alpha,
            interpretation=interpretation
        )
    
    except Exception as e:
        warnings.warn(f"ARCH test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="ARCH Test for Conditional Heteroskedasticity",
            test_statistic=np.nan,
            p_value=np.nan,
            critical_values={},
            is_stationary=False,
            alpha=alpha,
            interpretation=f"Test failed: {str(e)}"
        )


def white_test(
    timeseries: pd.Series,
    alpha: float = 0.05
) -> TestResult:
    """
    Perform White's test for heteroskedasticity.
    
    White's test regresses squared residuals on the original series and its square
    to detect non-constant variance patterns.
    
    Null Hypothesis: Homoskedasticity (constant variance)
    Alternative: Heteroskedasticity (non-constant variance)
    
    Args:
        timeseries: Input time series
        alpha: Significance level
        
    Returns:
        TestResult object with test statistics and interpretation
    """
    ts = timeseries.dropna().values
    n = len(ts)
    
    if n < 20:
        raise ValueError(
            f"Time series too short for White test. "
            f"Need at least 20 observations, got {n}"
        )
    
    try:
        # Create time index as predictor
        t = np.arange(n).reshape(-1, 1)
        
        # Fit simple linear model
        X = add_constant(t)
        model = OLS(ts, X).fit()
        residuals = model.resid
        
        # Regress squared residuals on predictors and their squares
        resid_sq = residuals ** 2
        X_white = np.column_stack([t, t**2])
        X_white = add_constant(X_white)
        
        white_model = OLS(resid_sq, X_white).fit()
        
        # Test statistic: n * R^2
        statistic = n * white_model.rsquared
        
        # Chi-square test with degrees of freedom = number of regressors (excluding constant)
        df = X_white.shape[1] - 1
        p_value = 1 - stats.chi2.cdf(statistic, df)
        
        is_stationary = p_value > alpha
        
        interpretation = (
            f"White's test regresses squared residuals on time and time-squared. "
            f"Test statistic (n*R²) = {statistic:.4f}, p-value = {p_value:.4f}. "
            f"With p-value {'>' if is_stationary else '<='} α = {alpha}, "
            f"we {'fail to reject' if is_stationary else 'reject'} the null hypothesis. "
            f"{'No significant' if is_stationary else 'Significant'} heteroskedasticity detected."
        )
        
        return TestResult(
            test_name="White's Test for Heteroskedasticity",
            test_statistic=statistic,
            p_value=p_value,
            critical_values={"df": df},
            is_stationary=is_stationary,
            alpha=alpha,
            interpretation=interpretation
        )
    
    except Exception as e:
        warnings.warn(f"White test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="White's Test for Heteroskedasticity",
            test_statistic=np.nan,
            p_value=np.nan,
            critical_values={},
            is_stationary=False,
            alpha=alpha,
            interpretation=f"Test failed: {str(e)}"
        )


def test_variance_stationarity(
    timeseries: pd.Series,
    method: str = "levene",
    alpha: float = 0.05,
    **kwargs
) -> TestResult:
    """
    Perform variance stationarity test using specified method.
    
    Args:
        timeseries: Input time series
        method: Test method ('levene', 'bartlett', 'white', 'arch')
        alpha: Significance level
        **kwargs: Additional arguments for specific tests
        
    Returns:
        TestResult object
        
    Raises:
        ValueError: If method is not recognized
    """
    method = method.lower()
    
    if method == "levene":
        return levene_test(timeseries, alpha, **kwargs)
    elif method == "bartlett":
        return bartlett_test(timeseries, alpha, **kwargs)
    elif method == "white":
        return white_test(timeseries, alpha)
    elif method == "arch":
        return arch_test(timeseries, alpha, **kwargs)
    else:
        raise ValueError(
            f"Unknown variance test method: {method}. "
            f"Valid options: 'levene', 'bartlett', 'white', 'arch'"
        )
