"""
Trend removal transformations.

These transformations help achieve constant mean (remove trend)
in time series data through differencing operations.
"""

from typing import Callable, Optional
import numpy as np
import pandas as pd
import warnings

from ..results import TransformationResult


def difference(
    timeseries: pd.Series,
    periods: int = 1,
    order: int = 1
) -> TransformationResult:
    """
    Apply differencing to remove trend.
    
    Differencing computes the change between consecutive observations,
    which can remove linear trends and make the series stationary.
    
    Args:
        timeseries: Input time series
        periods: Number of periods to shift for calculating differences
        order: Number of times to apply differencing
        
    Returns:
        TransformationResult with differenced data and inverse function
    """
    ts = timeseries.copy()
    original = ts.copy()
    
    # Store initial values for inverse transformation
    initial_values = []
    
    # Apply differencing 'order' times
    for i in range(order):
        initial_values.append(ts.iloc[:periods].values)
        ts = ts.diff(periods=periods).dropna()
    
    # Create inverse function
    def inverse(data: np.ndarray) -> np.ndarray:
        """Inverse differencing transformation."""
        result = pd.Series(data)
        
        # Apply inverse differencing in reverse order
        for i in range(order - 1, -1, -1):
            # Prepend initial values
            init_vals = initial_values[i]
            result = pd.concat([pd.Series(init_vals), result])
            
            # Cumulative sum to reverse differencing
            result = result.cumsum()
        
        return result.values
    
    # Calculate improvement (variance of differences should be smaller if trend removed)
    original_var = original.var()
    transformed_var = ts.var()
    improvement = (original_var - transformed_var) / original_var if original_var > 0 else 0
    
    return TransformationResult(
        transformation_name=f"Differencing (order={order}, periods={periods})",
        transformation_params={"periods": periods, "order": order, "initial_values": initial_values},
        original_data=original,
        transformed_data=ts,
        inverse_function=inverse,
        improvement_metric=improvement
    )


def seasonal_difference(
    timeseries: pd.Series,
    seasonal_period: int,
    order: int = 1
) -> TransformationResult:
    """
    Apply seasonal differencing to remove seasonal patterns.
    
    Seasonal differencing computes the change between observations
    separated by the seasonal period (e.g., 12 for monthly data with
    yearly seasonality).
    
    Args:
        timeseries: Input time series
        seasonal_period: Number of periods in a season
        order: Number of times to apply seasonal differencing
        
    Returns:
        TransformationResult with seasonally differenced data and inverse function
    """
    if seasonal_period < 2:
        raise ValueError(f"seasonal_period must be >= 2, got {seasonal_period}")
    
    ts = timeseries.copy()
    original = ts.copy()
    
    # Store initial values for inverse transformation
    initial_values = []
    
    # Apply seasonal differencing 'order' times
    for i in range(order):
        initial_values.append(ts.iloc[:seasonal_period].values)
        ts = ts.diff(periods=seasonal_period).dropna()
    
    # Create inverse function
    def inverse(data: np.ndarray) -> np.ndarray:
        """Inverse seasonal differencing transformation."""
        result = pd.Series(data)
        
        # Apply inverse differencing in reverse order
        for i in range(order - 1, -1, -1):
            # Prepend initial values
            init_vals = initial_values[i]
            result_list = list(init_vals)
            
            # Reconstruct by adding back seasonal differences
            for j, val in enumerate(result):
                if j + seasonal_period < len(result_list):
                    result_list.append(val + result_list[j])
                else:
                    result_list.append(val + result_list[-seasonal_period])
            
            result = pd.Series(result_list[seasonal_period:])
        
        return result.values
    
    # Calculate improvement
    original_var = original.var()
    transformed_var = ts.var()
    improvement = (original_var - transformed_var) / original_var if original_var > 0 else 0
    
    return TransformationResult(
        transformation_name=f"Seasonal Differencing (period={seasonal_period}, order={order})",
        transformation_params={
            "seasonal_period": seasonal_period,
            "order": order,
            "initial_values": initial_values
        },
        original_data=original,
        transformed_data=ts,
        inverse_function=inverse,
        improvement_metric=improvement
    )


def combined_difference(
    timeseries: pd.Series,
    seasonal_period: Optional[int] = None,
    trend_order: int = 1,
    seasonal_order: int = 0
) -> TransformationResult:
    """
    Apply both trend and seasonal differencing.
    
    This combines regular differencing (for trend) with seasonal differencing
    (for seasonality). Typically, seasonal differencing is applied first,
    then trend differencing.
    
    Args:
        timeseries: Input time series
        seasonal_period: Number of periods in a season (None to skip seasonal)
        trend_order: Order of trend differencing
        seasonal_order: Order of seasonal differencing
        
    Returns:
        TransformationResult with combined differencing
    """
    ts = timeseries.copy()
    original = ts.copy()
    transformations = []
    
    # Apply seasonal differencing first (if specified)
    if seasonal_period is not None and seasonal_order > 0:
        seasonal_result = seasonal_difference(ts, seasonal_period, seasonal_order)
        ts = seasonal_result.transformed_data
        transformations.append(("seasonal", seasonal_result))
    
    # Apply trend differencing
    if trend_order > 0:
        trend_result = difference(ts, periods=1, order=trend_order)
        ts = trend_result.transformed_data
        transformations.append(("trend", trend_result))
    
    # Create combined inverse function
    def inverse(data: np.ndarray) -> np.ndarray:
        """Inverse combined differencing."""
        result = data
        
        # Apply inverse transformations in reverse order
        for trans_type, trans_result in reversed(transformations):
            result = trans_result.inverse_function(result)
        
        return result
    
    # Calculate improvement
    original_var = original.var()
    transformed_var = ts.var()
    improvement = (original_var - transformed_var) / original_var if original_var > 0 else 0
    
    params = {
        "trend_order": trend_order,
        "seasonal_order": seasonal_order,
        "seasonal_period": seasonal_period
    }
    
    return TransformationResult(
        transformation_name=f"Combined Differencing (trend={trend_order}, seasonal={seasonal_order})",
        transformation_params=params,
        original_data=original,
        transformed_data=ts,
        inverse_function=inverse,
        improvement_metric=improvement
    )


def apply_trend_transformation(
    timeseries: pd.Series,
    method: str = "auto",
    seasonal_period: Optional[int] = None,
    test_func: Optional[Callable] = None,
    alpha: float = 0.05,
    max_order: int = 2
) -> TransformationResult:
    """
    Apply trend removal transformation, automatically selecting the best method.
    
    Args:
        timeseries: Input time series
        method: Transformation method ('difference', 'seasonal', 'combined', 'auto')
        seasonal_period: Seasonal period (required for seasonal/combined methods)
        test_func: Function to test trend stationarity (for 'auto' mode)
        alpha: Significance level for testing
        max_order: Maximum differencing order to try
        
    Returns:
        TransformationResult with best transformation
        
    Raises:
        ValueError: If method is not recognized or required parameters missing
    """
    method = method.lower()
    
    if method == "difference":
        return difference(timeseries, periods=1, order=1)
    
    elif method == "seasonal":
        if seasonal_period is None:
            raise ValueError("seasonal_period required for seasonal differencing")
        return seasonal_difference(timeseries, seasonal_period, order=1)
    
    elif method == "combined":
        if seasonal_period is None:
            raise ValueError("seasonal_period required for combined differencing")
        return combined_difference(timeseries, seasonal_period, trend_order=1, seasonal_order=1)
    
    elif method == "auto":
        # Try different differencing strategies
        candidates = []
        
        # Try simple differencing
        for order in range(1, max_order + 1):
            try:
                result = difference(timeseries, periods=1, order=order)
                candidates.append(result)
            except Exception as e:
                warnings.warn(f"Differencing order {order} failed: {e}")
        
        # Try seasonal differencing if period provided
        if seasonal_period is not None:
            try:
                result = seasonal_difference(timeseries, seasonal_period, order=1)
                candidates.append(result)
            except Exception as e:
                warnings.warn(f"Seasonal differencing failed: {e}")
            
            # Try combined
            try:
                result = combined_difference(timeseries, seasonal_period, trend_order=1, seasonal_order=1)
                candidates.append(result)
            except Exception as e:
                warnings.warn(f"Combined differencing failed: {e}")
        
        if not candidates:
            raise ValueError("All differencing methods failed")
        
        # If test function provided, use it to select best
        if test_func is not None:
            best_result = None
            best_p_value = -1
            
            for result in candidates:
                try:
                    test_result = test_func(result.transformed_data, alpha)
                    # For trend tests, we want p-value < alpha (reject non-stationarity)
                    # So we look for the transformation that gives us stationarity
                    if test_result.is_stationary and test_result.p_value > best_p_value:
                        best_p_value = test_result.p_value
                        best_result = result
                except Exception as e:
                    warnings.warn(f"Test failed for {result.transformation_name}: {e}")
            
            if best_result is not None:
                return best_result
        
        # Otherwise, select based on improvement metric (prefer simpler transformations)
        # Sort by improvement, but penalize higher order
        best_result = max(
            candidates,
            key=lambda x: x.improvement_metric - 0.1 * x.transformation_params.get('order', 0)
        )
        return best_result
    
    elif method == "none":
        # No transformation
        return TransformationResult(
            transformation_name="None",
            transformation_params={},
            original_data=timeseries,
            transformed_data=timeseries.copy(),
            inverse_function=lambda x: x,
            improvement_metric=0.0
        )
    
    else:
        raise ValueError(
            f"Unknown trend transformation method: {method}. "
            f"Valid options: 'difference', 'seasonal', 'combined', 'auto', 'none'"
        )
