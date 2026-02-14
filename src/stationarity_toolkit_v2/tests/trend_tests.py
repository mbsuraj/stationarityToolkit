"""
Trend stationarity tests.

This module provides tests for detecting unit roots and trend non-stationarity
in time series data. These tests check whether the mean of the series is constant
over time.
"""

from typing import Dict
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
import warnings

from ..results import TestResult


def adf_test(
    timeseries: pd.Series,
    alpha: float = 0.05,
    regression: str = "c",
    autolag: str = "AIC"
) -> TestResult:
    """
    Perform Augmented Dickey-Fuller (ADF) test for unit root.
    
    The ADF test is one of the most widely used tests for trend stationarity.
    It tests whether a unit root is present in the time series.
    
    Null Hypothesis: Unit root is present (series is non-stationary)
    Alternative: No unit root (series is stationary)
    
    Args:
        timeseries: Input time series
        alpha: Significance level
        regression: Type of regression ('c' = constant, 'ct' = constant and trend, 'n' = no constant/trend)
        autolag: Method to select lag length ('AIC', 'BIC', 't-stat', or None)
        
    Returns:
        TestResult object with test statistics and interpretation
    """
    ts = timeseries.dropna().values
    
    if len(ts) < 10:
        raise ValueError(
            f"Time series too short for ADF test. "
            f"Need at least 10 observations, got {len(ts)}"
        )
    
    try:
        # Perform ADF test
        result = adfuller(ts, regression=regression, autolag=autolag)
        
        test_statistic = result[0]
        p_value = result[1]
        n_lags = result[2]
        n_obs = result[3]
        critical_values = result[4]
        
        # ADF: reject null (non-stationary) if p-value < alpha
        is_stationary = p_value < alpha
        
        regression_desc = {
            'c': 'constant only',
            'ct': 'constant and trend',
            'ctt': 'constant, linear and quadratic trend',
            'n': 'no constant or trend'
        }
        
        interpretation = (
            f"ADF test with {regression_desc.get(regression, regression)} regression. "
            f"Test statistic = {test_statistic:.4f}, p-value = {p_value:.4f}. "
            f"Used {n_lags} lags (selected by {autolag}) with {n_obs} observations. "
            f"With p-value {'<' if is_stationary else '>='} α = {alpha}, "
            f"we {'reject' if is_stationary else 'fail to reject'} the null hypothesis. "
            f"The series appears to be {'stationary' if is_stationary else 'non-stationary'} "
            f"(unit root {'not' if is_stationary else ''} present)."
        )
        
        return TestResult(
            test_name="Augmented Dickey-Fuller (ADF) Test",
            test_statistic=test_statistic,
            p_value=p_value,
            critical_values=critical_values,
            is_stationary=is_stationary,
            alpha=alpha,
            interpretation=interpretation
        )
    
    except Exception as e:
        warnings.warn(f"ADF test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="Augmented Dickey-Fuller (ADF) Test",
            test_statistic=np.nan,
            p_value=np.nan,
            critical_values={},
            is_stationary=False,
            alpha=alpha,
            interpretation=f"Test failed: {str(e)}"
        )


def kpss_test(
    timeseries: pd.Series,
    alpha: float = 0.05,
    regression: str = "c",
    nlags: str = "auto"
) -> TestResult:
    """
    Perform Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity.
    
    The KPSS test has the opposite null hypothesis compared to ADF.
    It's often used in conjunction with ADF for a more robust analysis.
    
    Null Hypothesis: Series is stationary
    Alternative: Series is non-stationary (unit root present)
    
    Args:
        timeseries: Input time series
        alpha: Significance level
        regression: Type of regression ('c' = level stationary, 'ct' = trend stationary)
        nlags: Number of lags ('auto' or integer)
        
    Returns:
        TestResult object with test statistics and interpretation
    """
    ts = timeseries.dropna().values
    
    if len(ts) < 10:
        raise ValueError(
            f"Time series too short for KPSS test. "
            f"Need at least 10 observations, got {len(ts)}"
        )
    
    try:
        # Perform KPSS test
        result = kpss(ts, regression=regression, nlags=nlags)
        
        test_statistic = result[0]
        p_value = result[1]
        n_lags = result[2]
        critical_values = result[3]
        
        # KPSS: reject null (stationary) if p-value < alpha
        is_stationary = p_value > alpha
        
        regression_desc = {
            'c': 'level stationarity',
            'ct': 'trend stationarity'
        }
        
        interpretation = (
            f"KPSS test for {regression_desc.get(regression, regression)}. "
            f"Test statistic = {test_statistic:.4f}, p-value = {p_value:.4f}. "
            f"Used {n_lags} lags. "
            f"With p-value {'>' if is_stationary else '<='} α = {alpha}, "
            f"we {'fail to reject' if is_stationary else 'reject'} the null hypothesis. "
            f"The series appears to be {'stationary' if is_stationary else 'non-stationary'}. "
            f"Note: KPSS has opposite null hypothesis compared to ADF."
        )
        
        return TestResult(
            test_name="Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test",
            test_statistic=test_statistic,
            p_value=p_value,
            critical_values=critical_values,
            is_stationary=is_stationary,
            alpha=alpha,
            interpretation=interpretation
        )
    
    except Exception as e:
        warnings.warn(f"KPSS test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test",
            test_statistic=np.nan,
            p_value=np.nan,
            critical_values={},
            is_stationary=False,
            alpha=alpha,
            interpretation=f"Test failed: {str(e)}"
        )


def phillips_perron_test(
    timeseries: pd.Series,
    alpha: float = 0.05,
    trend: str = "c",
    lags: int = None
) -> TestResult:
    """
    Perform Phillips-Perron (PP) test for unit root.
    
    The PP test is similar to ADF but uses a different method to handle
    serial correlation. It tests for TREND stationarity (unit roots),
    NOT variance stationarity.
    
    NOTE: This test is for TREND stationarity, not variance stationarity!
    
    Null Hypothesis: Unit root is present (series is non-stationary)
    Alternative: No unit root (series is stationary)
    
    Args:
        timeseries: Input time series
        alpha: Significance level
        trend: Type of trend ('c' = constant, 'ct' = constant and trend, 'n' = no trend)
        lags: Number of lags (None for automatic selection)
        
    Returns:
        TestResult object with test statistics and interpretation
    """
    ts = timeseries.dropna().values
    
    if len(ts) < 10:
        raise ValueError(
            f"Time series too short for Phillips-Perron test. "
            f"Need at least 10 observations, got {len(ts)}"
        )
    
    try:
        # Perform Phillips-Perron test
        pp = PhillipsPerron(ts, trend=trend, lags=lags)
        
        test_statistic = pp.stat
        p_value = pp.pvalue
        critical_values = pp.critical_values
        
        # PP: reject null (non-stationary) if p-value < alpha
        is_stationary = p_value < alpha
        
        trend_desc = {
            'c': 'constant only',
            'ct': 'constant and trend',
            'n': 'no constant or trend'
        }
        
        interpretation = (
            f"Phillips-Perron test with {trend_desc.get(trend, trend)}. "
            f"Test statistic = {test_statistic:.4f}, p-value = {p_value:.4f}. "
            f"With p-value {'<' if is_stationary else '>='} α = {alpha}, "
            f"we {'reject' if is_stationary else 'fail to reject'} the null hypothesis. "
            f"The series appears to be {'stationary' if is_stationary else 'non-stationary'} "
            f"(unit root {'not' if is_stationary else ''} present). "
            f"NOTE: This tests TREND stationarity, not variance stationarity!"
        )
        
        return TestResult(
            test_name="Phillips-Perron (PP) Test for Unit Root",
            test_statistic=test_statistic,
            p_value=p_value,
            critical_values=critical_values,
            is_stationary=is_stationary,
            alpha=alpha,
            interpretation=interpretation
        )
    
    except Exception as e:
        warnings.warn(f"Phillips-Perron test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="Phillips-Perron (PP) Test for Unit Root",
            test_statistic=np.nan,
            p_value=np.nan,
            critical_values={},
            is_stationary=False,
            alpha=alpha,
            interpretation=f"Test failed: {str(e)}"
        )


def combined_trend_test(
    timeseries: pd.Series,
    alpha: float = 0.05
) -> TestResult:
    """
    Perform combined ADF and KPSS test for more robust trend stationarity assessment.
    
    Uses both tests to provide a more reliable conclusion:
    - If both agree on stationary: High confidence it's stationary
    - If both agree on non-stationary: High confidence it's non-stationary
    - If they disagree: Inconclusive, may need differencing
    
    Args:
        timeseries: Input time series
        alpha: Significance level
        
    Returns:
        TestResult object with combined interpretation
    """
    adf_result = adf_test(timeseries, alpha)
    kpss_result = kpss_test(timeseries, alpha)
    
    # Determine combined result
    if adf_result.is_stationary and kpss_result.is_stationary:
        is_stationary = True
        conclusion = "Both ADF and KPSS agree: series is stationary"
        confidence = "High"
    elif not adf_result.is_stationary and not kpss_result.is_stationary:
        is_stationary = False
        conclusion = "Both ADF and KPSS agree: series is non-stationary"
        confidence = "High"
    else:
        is_stationary = False
        conclusion = "ADF and KPSS disagree: results are inconclusive, differencing recommended"
        confidence = "Low"
    
    interpretation = (
        f"Combined ADF-KPSS Test:\n"
        f"  ADF: {'Stationary' if adf_result.is_stationary else 'Non-stationary'} "
        f"(p={adf_result.p_value:.4f})\n"
        f"  KPSS: {'Stationary' if kpss_result.is_stationary else 'Non-stationary'} "
        f"(p={kpss_result.p_value:.4f})\n"
        f"Conclusion: {conclusion}\n"
        f"Confidence: {confidence}"
    )
    
    return TestResult(
        test_name="Combined ADF-KPSS Test",
        test_statistic=adf_result.test_statistic,
        p_value=adf_result.p_value,
        critical_values={
            "ADF": adf_result.critical_values,
            "KPSS": kpss_result.critical_values
        },
        is_stationary=is_stationary,
        alpha=alpha,
        interpretation=interpretation
    )


def test_trend_stationarity(
    timeseries: pd.Series,
    method: str = "adf",
    alpha: float = 0.05,
    **kwargs
) -> TestResult:
    """
    Perform trend stationarity test using specified method.
    
    Args:
        timeseries: Input time series
        method: Test method ('adf', 'kpss', 'pp', 'combined')
        alpha: Significance level
        **kwargs: Additional arguments for specific tests
        
    Returns:
        TestResult object
        
    Raises:
        ValueError: If method is not recognized
    """
    method = method.lower()
    
    if method == "adf":
        return adf_test(timeseries, alpha, **kwargs)
    elif method == "kpss":
        return kpss_test(timeseries, alpha, **kwargs)
    elif method == "pp" or method == "phillips_perron":
        return phillips_perron_test(timeseries, alpha, **kwargs)
    elif method == "combined":
        return combined_trend_test(timeseries, alpha)
    else:
        raise ValueError(
            f"Unknown trend test method: {method}. "
            f"Valid options: 'adf', 'kpss', 'pp', 'combined'"
        )
