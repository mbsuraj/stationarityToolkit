"""Trend stationarity tests."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

from ..results import TestResult


def adf_test(
    timeseries: pd.Series,
    alpha: float = 0.05,
    autolag: str = "AIC"
) -> TestResult:
    """
    Perform Augmented Dickey-Fuller (ADF) test for unit root.
    
    Runs both 'c' (constant) and 'ct' (constant+trend) specifications to distinguish
    between unit roots and deterministic trends.
    
    Null Hypothesis: Unit root is present (series is non-stationary)
    Alternative: No unit root (series is stationary)
    
    Args:
        timeseries: Input time series
        alpha: Significance level
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
        result_c = adfuller(ts, regression='c', autolag=autolag)
        result_ct = adfuller(ts, regression='ct', autolag=autolag)
        
        c_stationary = result_c[1] < alpha
        ct_stationary = result_ct[1] < alpha
        
        if c_stationary and ct_stationary:
            is_stationary = True
            educational_note = "Stationary around constant mean"
            interpretation = f"H0: Unit root. ADF-c p={result_c[1]:.4f} < {alpha}, ADF-ct p={result_ct[1]:.4f} < {alpha}. Reject H0."
        elif c_stationary and not ct_stationary:
            is_stationary = True
            educational_note = "Stationary around constant mean (ct test lost power)"
            interpretation = f"H0: Unit root. ADF-c p={result_c[1]:.4f} < {alpha}, ADF-ct p={result_ct[1]:.4f} >= {alpha}. Reject H0 (c mode)."
        elif not c_stationary and ct_stationary:
            is_stationary = False
            educational_note = "Deterministic trend detected - stationary after detrending"
            interpretation = f"H0: Unit root. ADF-c p={result_c[1]:.4f} >= {alpha}, ADF-ct p={result_ct[1]:.4f} < {alpha}. Deterministic trend."
        else:
            is_stationary = False
            educational_note = "Unit root detected - requires differencing"
            interpretation = f"H0: Unit root. ADF-c p={result_c[1]:.4f} >= {alpha}, ADF-ct p={result_ct[1]:.4f} >= {alpha}. Fail to reject H0."
        
        return TestResult(
            test_name="Augmented Dickey-Fuller (ADF) Test",
            statistic=result_c[0],
            p_value=result_c[1],
            is_stationary=is_stationary,
            interpretation=interpretation,
            educational_note=educational_note
        )
    
    except Exception as e:
        warnings.warn(f"ADF test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="Augmented Dickey-Fuller (ADF) Test",
            statistic=np.nan,
            p_value=np.nan,
            is_stationary=False,
            interpretation=f"Test failed: {str(e)}",
            educational_note="Test execution failed."
        )


def kpss_test(
    timeseries: pd.Series,
    alpha: float = 0.05,
    nlags: str = "auto"
) -> TestResult:
    """
    Perform Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity.
    
    Runs both 'c' (level) and 'ct' (trend) specifications to distinguish
    between unit roots and deterministic trends.
    
    KPSS has opposite null hypothesis compared to ADF.
    
    Null Hypothesis: Series is stationary
    Alternative: Series is non-stationary (unit root present)
    
    Args:
        timeseries: Input time series
        alpha: Significance level
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
        result_c = kpss(ts, regression='c', nlags=nlags)
        result_ct = kpss(ts, regression='ct', nlags=nlags)
        
        c_stationary = result_c[1] > alpha
        ct_stationary = result_ct[1] > alpha
        
        if c_stationary and ct_stationary:
            is_stationary = True
            educational_note = "Stationary around constant mean"
            interpretation = f"H0: Stationary. KPSS-c p={result_c[1]:.4f} > {alpha}, KPSS-ct p={result_ct[1]:.4f} > {alpha}. Fail to reject H0."
        elif c_stationary and not ct_stationary:
            is_stationary = True
            educational_note = "Stationary around constant mean (ct test lost power)"
            interpretation = f"H0: Stationary. KPSS-c p={result_c[1]:.4f} > {alpha}, KPSS-ct p={result_ct[1]:.4f} <= {alpha}. Fail to reject H0 (c mode)."
        elif not c_stationary and ct_stationary:
            is_stationary = False
            educational_note = "Deterministic trend detected - stationary after detrending"
            interpretation = f"H0: Stationary. KPSS-c p={result_c[1]:.4f} <= {alpha}, KPSS-ct p={result_ct[1]:.4f} > {alpha}. Deterministic trend."
        else:
            is_stationary = False
            educational_note = "Unit root detected - requires differencing"
            interpretation = f"H0: Stationary. KPSS-c p={result_c[1]:.4f} <= {alpha}, KPSS-ct p={result_ct[1]:.4f} <= {alpha}. Reject H0."
        
        return TestResult(
            test_name="Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test",
            statistic=result_c[0],
            p_value=result_c[1],
            is_stationary=is_stationary,
            interpretation=interpretation,
            educational_note=educational_note
        )
    
    except Exception as e:
        warnings.warn(f"KPSS test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test",
            statistic=np.nan,
            p_value=np.nan,
            is_stationary=False,
            interpretation=f"Test failed: {str(e)}",
            educational_note="Test execution failed."
        )


def phillips_perron_test(
    timeseries: pd.Series,
    alpha: float = 0.05,
    lags: int = None
) -> TestResult:
    """
    Perform Phillips-Perron (PP) test for unit root.
    
    Runs both 'c' (constant) and 'ct' (constant+trend) specifications to distinguish
    between unit roots and deterministic trends.
    
    PP is similar to ADF but uses non-parametric correction for serial correlation.
    
    Null Hypothesis: Unit root is present (series is non-stationary)
    Alternative: No unit root (series is stationary)
    
    Args:
        timeseries: Input time series
        alpha: Significance level
        lags: Number of lags (None for automatic selection)
        
    Returns:
        TestResult object with test statistics and interpretation
    """
    from arch.unitroot import PhillipsPerron
    
    ts = timeseries.dropna().values
    
    if len(ts) < 10:
        raise ValueError(
            f"Time series too short for Phillips-Perron test. "
            f"Need at least 10 observations, got {len(ts)}"
        )
    
    try:
        pp_c = PhillipsPerron(ts, trend='c', lags=lags)
        pp_ct = PhillipsPerron(ts, trend='ct', lags=lags)
        
        c_stationary = pp_c.pvalue < alpha
        ct_stationary = pp_ct.pvalue < alpha
        
        if c_stationary and ct_stationary:
            is_stationary = True
            educational_note = "Stationary around constant mean"
            interpretation = f"H0: Unit root. PP-c p={pp_c.pvalue:.4f} < {alpha}, PP-ct p={pp_ct.pvalue:.4f} < {alpha}. Reject H0."
        elif c_stationary and not ct_stationary:
            is_stationary = True
            educational_note = "Stationary around constant mean (ct test lost power)"
            interpretation = f"H0: Unit root. PP-c p={pp_c.pvalue:.4f} < {alpha}, PP-ct p={pp_ct.pvalue:.4f} >= {alpha}. Reject H0 (c mode)."
        elif not c_stationary and ct_stationary:
            is_stationary = False
            educational_note = "Deterministic trend detected - stationary after detrending"
            interpretation = f"H0: Unit root. PP-c p={pp_c.pvalue:.4f} >= {alpha}, PP-ct p={pp_ct.pvalue:.4f} < {alpha}. Deterministic trend."
        else:
            is_stationary = False
            educational_note = "Unit root detected - requires differencing"
            interpretation = f"H0: Unit root. PP-c p={pp_c.pvalue:.4f} >= {alpha}, PP-ct p={pp_ct.pvalue:.4f} >= {alpha}. Fail to reject H0."
        
        return TestResult(
            test_name="Phillips-Perron (PP) Test",
            statistic=pp_c.stat,
            p_value=pp_c.pvalue,
            is_stationary=is_stationary,
            interpretation=interpretation,
            educational_note=educational_note
        )
    
    except Exception as e:
        warnings.warn(f"Phillips-Perron test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="Phillips-Perron (PP) Test",
            statistic=np.nan,
            p_value=np.nan,
            is_stationary=False,
            interpretation=f"Test failed: {str(e)}",
            educational_note="Test execution failed."
        )


def zivot_andrews_test(
    timeseries: pd.Series,
    alpha: float = 0.05,
    maxlag: int = None
) -> TestResult:
    """
    Perform Zivot-Andrews test for structural breaks in trend.

    Detects discrete structural breaks (regime changes, level/trend shifts).
    Does NOT detect smooth trends - use ADF/PP/KPSS for that.
    May find spurious breaks in pure noise.

    Null Hypothesis: Unit root with no structural break
    Alternative: Trend stationary with structural break

    Args:
        timeseries: Input time series
        alpha: Significance level
        maxlag: Maximum number of lags (None for automatic selection)

    Returns:
        TestResult object with test statistics and interpretation
    """
    from statsmodels.tsa.stattools import zivot_andrews
    
    ts = timeseries.dropna()
    
    if len(ts) < 20:
        raise ValueError(
            f"Time series too short for Zivot-Andrews test. "
            f"Need at least 20 observations, got {len(ts)}"
        )
    
    try:
        results = {}
        for trend_type in ['c', 't', 'ct']:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                za_result = zivot_andrews(ts, maxlag=maxlag, regression=trend_type)
                
                results[trend_type] = {
                    'stat': float(za_result[0]),
                    'pvalue': float(za_result[1]),
                    'breakpoint': int(za_result[4])
                }
        
        # Check which tests reject null (p < alpha means reject unit root = stationary)
        c_reject = results['c']['pvalue'] < alpha
        t_reject = results['t']['pvalue'] < alpha
        ct_reject = results['ct']['pvalue'] < alpha
        
        # If any test rejects unit root, series is stationary (with possible break)
        if c_reject or t_reject or ct_reject:
            is_stationary = True
            breaks_detected = []
            if c_reject:
                breaks_detected.append(f"level shift at obs {results['c']['breakpoint']}")
            if t_reject:
                breaks_detected.append(f"trend shift at obs {results['t']['breakpoint']}")
            if ct_reject:
                breaks_detected.append(f"level+trend shift at obs {results['ct']['breakpoint']}")
            
            educational_note = f"Stationary with structural breaks: {', '.join(breaks_detected)}. Note: ZA detects discrete breaks, not smooth trends. Breaks may be spurious in noise."
            
            p_vals = []
            if c_reject:
                p_vals.append(f"ZA-c p={results['c']['pvalue']:.4f}")
            if t_reject:
                p_vals.append(f"ZA-t p={results['t']['pvalue']:.4f}")
            if ct_reject:
                p_vals.append(f"ZA-ct p={results['ct']['pvalue']:.4f}")
            
            interpretation = f"H0: Unit root with no break. {', '.join(p_vals)} < {alpha}. Reject H0 - stationary."
            
            # Use the most significant result
            min_p = min(results['c']['pvalue'], results['t']['pvalue'], results['ct']['pvalue'])
            for trend_type, res in results.items():
                if res['pvalue'] == min_p:
                    test_statistic = res['stat']
                    p_value = res['pvalue']
                    break
        else:
            is_stationary = False
            educational_note = "Unit root detected - consider differencing"
            interpretation = (
                f"H0: Unit root with no break. "
                f"ZA-c p={results['c']['pvalue']:.4f}, "
                f"ZA-t p={results['t']['pvalue']:.4f}, "
                f"ZA-ct p={results['ct']['pvalue']:.4f} (all >= {alpha}). "
                f"Fail to reject H0."
            )
            test_statistic = results['c']['stat']
            p_value = results['c']['pvalue']
        
        return TestResult(
            test_name="Zivot-Andrews Test for Structural Breaks",
            statistic=test_statistic,
            p_value=p_value,
            is_stationary=is_stationary,
            interpretation=interpretation,
            educational_note=educational_note
        )
    
    except Exception as e:
        return TestResult(
            test_name="Zivot-Andrews Test for Structural Breaks",
            statistic=np.nan,
            p_value=np.nan,
            is_stationary=False,
            interpretation=f"Test failed: {str(e)}",
            educational_note="Test execution failed."
        )



def run_all_trend_tests(
    timeseries: pd.Series,
    alpha: float = 0.05
) -> list[TestResult]:
    """
    Run all trend tests and return list of TestResult objects.
    
    Runs: ADF, KPSS, Phillips-Perron, and Zivot-Andrews tests.
    
    Args:
        timeseries: Input time series
        alpha: Significance level for all tests
        
    Returns:
        List of TestResult objects from all trend tests
    """
    results = []
    
    results.append(adf_test(timeseries, alpha=alpha))
    results.append(kpss_test(timeseries, alpha=alpha))
    results.append(phillips_perron_test(timeseries, alpha=alpha))
    results.append(zivot_andrews_test(timeseries, alpha=alpha))
    
    return results
