"""Variance stationarity tests."""

import numpy as np
import pandas as pd
from scipy import stats
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
    
    if is_stationary:
        educational_note = "Constant variance across time"
        interpretation = f"H0: Equal variances across segments. Levene p={p_value:.4f} > {alpha}. Fail to reject H0."
    else:
        educational_note = "Variance changes detected - consider Box-Cox or Yeo-Johnson transform"
        interpretation = f"H0: Equal variances across segments. Levene p={p_value:.4f} <= {alpha}. Reject H0."
    
    return TestResult(
        test_name="Levene's Test for Variance Homogeneity",
        statistic=statistic,
        p_value=p_value,
        is_stationary=is_stationary,
        interpretation=interpretation,
        educational_note=educational_note
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
        ts[i*window_size:(i+1)*window_size].astype(np.float64)
        for i in range(n_segments)
    ]
    
    try:
        # Perform Bartlett's test
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            statistic, p_value = stats.bartlett(*segments)
        
        is_stationary = p_value > alpha
        
        if is_stationary:
            educational_note = "Constant variance across time"
            interpretation = f"H0: Equal variances across segments (assumes normality). Bartlett p={p_value:.4f} > {alpha}. Fail to reject H0."
        else:
            educational_note = "Variance changes detected - consider Box-Cox or Yeo-Johnson transform"
            interpretation = f"H0: Equal variances across segments (assumes normality). Bartlett p={p_value:.4f} <= {alpha}. Reject H0."
        
        return TestResult(
            test_name="Bartlett's Test for Variance Homogeneity",
            statistic=statistic,
            p_value=p_value,
            is_stationary=is_stationary,
            interpretation=interpretation,
            educational_note=educational_note
        )
        
    except Exception as e:
        return TestResult(
            test_name="Bartlett's Test for Variance Homogeneity",
            statistic=np.nan,
            p_value=np.nan,
            is_stationary=False,
            interpretation=f"Test failed: {str(e)}",
            educational_note="Test execution failed."
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
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    import warnings
    
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
        
        if is_stationary:
            educational_note = "Constant variance across time"
            interpretation = f"H0: Homoskedasticity. White p={p_value:.4f} > {alpha}. Fail to reject H0."
        else:
            educational_note = "Time-dependent variance detected - consider Box-Cox or Yeo-Johnson transform"
            interpretation = f"H0: Homoskedasticity. White p={p_value:.4f} <= {alpha}. Reject H0."
        
        return TestResult(
            test_name="White's Test for Heteroskedasticity",
            statistic=statistic,
            p_value=p_value,
            is_stationary=is_stationary,
            interpretation=interpretation,
            educational_note=educational_note
        )
    
    except Exception as e:
        warnings.warn(f"White test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="White's Test for Heteroskedasticity",
            statistic=np.nan,
            p_value=np.nan,
            is_stationary=False,
            interpretation=f"Test failed: {str(e)}",
            educational_note=""
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
    from statsmodels.stats.diagnostic import het_arch
    import warnings
    
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
        
        if is_stationary:
            educational_note = "Constant variance across time"
            interpretation = f"H0: No ARCH effects. ARCH p={lm_pvalue:.4f} > {alpha}. Fail to reject H0."
        else:
            educational_note = "Volatility clustering detected - consider GARCH modeling for variance (models time-varying variance)"
            interpretation = f"H0: No ARCH effects. ARCH p={lm_pvalue:.4f} <= {alpha}. Reject H0."
        
        return TestResult(
            test_name="ARCH Test for Conditional Heteroskedasticity",
            statistic=lm_statistic,
            p_value=lm_pvalue,
            is_stationary=is_stationary,
            interpretation=interpretation,
            educational_note=educational_note
        )
    
    except Exception as e:
        warnings.warn(f"ARCH test failed: {str(e)}. Returning inconclusive result.")
        return TestResult(
            test_name="ARCH Test for Conditional Heteroskedasticity",
            statistic=np.nan,
            p_value=np.nan,
            is_stationary=False,
            interpretation=f"Test failed: {str(e)}",
            educational_note=""
        )




def run_all_variance_tests(
    timeseries: pd.Series,
    alpha: float = 0.05
) -> list[TestResult]:
    """
    Run all variance tests and return results.

    Runs: Levene, Bartlett, White, and ARCH tests.

    Args:
        timeseries: Input time series
        alpha: Significance level

    Returns:
        List of TestResult objects from all variance tests
    """
    results = []

    results.append(levene_test(timeseries, alpha))
    results.append(bartlett_test(timeseries, alpha))
    results.append(white_test(timeseries, alpha))
    results.append(arch_test(timeseries, alpha))

    return results
