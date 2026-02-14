"""
Variance stabilizing transformations.

These transformations help achieve constant variance (homoskedasticity)
in time series data.
"""

from typing import Callable, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import boxcox
import warnings

from ..results import TransformationResult


def log_transform(
    timeseries: pd.Series,
    offset: float = None
) -> TransformationResult:
    """
    Apply logarithmic transformation to stabilize variance.
    
    Log transformation is effective when variance increases proportionally
    with the level of the series. Requires positive values.
    
    Args:
        timeseries: Input time series
        offset: Constant to add before log (auto-calculated if None)
        
    Returns:
        TransformationResult with transformed data and inverse function
    """
    ts = timeseries.copy()
    
    # Calculate offset if needed
    if offset is None:
        min_val = ts.min()
        if min_val <= 0:
            offset = abs(min_val) + 1
            warnings.warn(
                f"Series contains non-positive values. Adding offset of {offset:.4f}"
            )
        else:
            offset = 0
    
    # Apply transformation
    transformed = np.log(ts + offset)
    
    # Create inverse function
    def inverse(data: np.ndarray) -> np.ndarray:
        """Inverse log transformation."""
        return np.exp(data) - offset
    
    # Calculate improvement (variance reduction)
    original_var = ts.var()
    transformed_var = transformed.var()
    improvement = (original_var - transformed_var) / original_var if original_var > 0 else 0
    
    return TransformationResult(
        transformation_name="Log Transform",
        transformation_params={"offset": offset},
        original_data=ts,
        transformed_data=transformed,
        inverse_function=inverse,
        improvement_metric=improvement
    )


def sqrt_transform(
    timeseries: pd.Series,
    offset: float = None
) -> TransformationResult:
    """
    Apply square root transformation to stabilize variance.
    
    Square root transformation is effective for count data or when
    variance increases with the mean. Requires non-negative values.
    
    Args:
        timeseries: Input time series
        offset: Constant to add before sqrt (auto-calculated if None)
        
    Returns:
        TransformationResult with transformed data and inverse function
    """
    ts = timeseries.copy()
    
    # Calculate offset if needed
    if offset is None:
        min_val = ts.min()
        if min_val < 0:
            offset = abs(min_val)
            warnings.warn(
                f"Series contains negative values. Adding offset of {offset:.4f}"
            )
        else:
            offset = 0
    
    # Apply transformation
    transformed = np.sqrt(ts + offset)
    
    # Create inverse function
    def inverse(data: np.ndarray) -> np.ndarray:
        """Inverse square root transformation."""
        return data ** 2 - offset
    
    # Calculate improvement
    original_var = ts.var()
    transformed_var = transformed.var()
    improvement = (original_var - transformed_var) / original_var if original_var > 0 else 0
    
    return TransformationResult(
        transformation_name="Square Root Transform",
        transformation_params={"offset": offset},
        original_data=ts,
        transformed_data=transformed,
        inverse_function=inverse,
        improvement_metric=improvement
    )


def boxcox_transform(
    timeseries: pd.Series,
    lmbda: float = None,
    offset: float = None
) -> TransformationResult:
    """
    Apply Box-Cox transformation to stabilize variance.
    
    Box-Cox is a family of power transformations that includes log (λ=0)
    and square root (λ=0.5) as special cases. It automatically finds the
    optimal λ parameter. Requires positive values.
    
    Args:
        timeseries: Input time series
        lmbda: Lambda parameter (auto-optimized if None)
        offset: Constant to add before transformation (auto-calculated if None)
        
    Returns:
        TransformationResult with transformed data and inverse function
    """
    ts = timeseries.copy()
    
    # Calculate offset if needed
    if offset is None:
        min_val = ts.min()
        if min_val <= 0:
            offset = abs(min_val) + 1
            warnings.warn(
                f"Series contains non-positive values. Adding offset of {offset:.4f}"
            )
        else:
            offset = 0
    
    # Apply Box-Cox transformation
    ts_positive = ts + offset
    
    if lmbda is None:
        # Optimize lambda
        transformed, fitted_lambda = boxcox(ts_positive)
        lmbda = fitted_lambda
    else:
        # Use provided lambda
        if lmbda == 0:
            transformed = np.log(ts_positive)
        else:
            transformed = (ts_positive ** lmbda - 1) / lmbda
    
    transformed = pd.Series(transformed, index=ts.index)
    
    # Create inverse function
    def inverse(data: np.ndarray) -> np.ndarray:
        """Inverse Box-Cox transformation."""
        if lmbda == 0:
            return np.exp(data) - offset
        else:
            return np.power(data * lmbda + 1, 1 / lmbda) - offset
    
    # Calculate improvement
    original_var = ts.var()
    transformed_var = transformed.var()
    improvement = (original_var - transformed_var) / original_var if original_var > 0 else 0
    
    return TransformationResult(
        transformation_name="Box-Cox Transform",
        transformation_params={"lambda": lmbda, "offset": offset},
        original_data=ts,
        transformed_data=transformed,
        inverse_function=inverse,
        improvement_metric=improvement
    )


def apply_variance_transformation(
    timeseries: pd.Series,
    method: str = "auto",
    test_func: Optional[Callable] = None,
    alpha: float = 0.05
) -> TransformationResult:
    """
    Apply variance stabilizing transformation, automatically selecting the best method.
    
    Args:
        timeseries: Input time series
        method: Transformation method ('log', 'sqrt', 'boxcox', 'auto')
        test_func: Function to test variance stationarity (for 'auto' mode)
        alpha: Significance level for testing
        
    Returns:
        TransformationResult with best transformation
        
    Raises:
        ValueError: If method is not recognized
    """
    method = method.lower()
    
    if method == "log":
        return log_transform(timeseries)
    elif method == "sqrt":
        return sqrt_transform(timeseries)
    elif method == "boxcox":
        return boxcox_transform(timeseries)
    elif method == "auto":
        # Try all methods and select best
        methods = []
        
        try:
            methods.append(log_transform(timeseries))
        except Exception as e:
            warnings.warn(f"Log transform failed: {e}")
        
        try:
            methods.append(sqrt_transform(timeseries))
        except Exception as e:
            warnings.warn(f"Sqrt transform failed: {e}")
        
        try:
            methods.append(boxcox_transform(timeseries))
        except Exception as e:
            warnings.warn(f"Box-Cox transform failed: {e}")
        
        if not methods:
            raise ValueError("All transformation methods failed")
        
        # If test function provided, use it to select best
        if test_func is not None:
            best_result = None
            best_p_value = -1
            
            for result in methods:
                try:
                    test_result = test_func(result.transformed_data, alpha)
                    if test_result.p_value > best_p_value:
                        best_p_value = test_result.p_value
                        best_result = result
                except Exception as e:
                    warnings.warn(f"Test failed for {result.transformation_name}: {e}")
            
            if best_result is not None:
                return best_result
        
        # Otherwise, select based on variance reduction
        best_result = max(methods, key=lambda x: x.improvement_metric)
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
            f"Unknown variance transformation method: {method}. "
            f"Valid options: 'log', 'sqrt', 'boxcox', 'auto', 'none'"
        )
