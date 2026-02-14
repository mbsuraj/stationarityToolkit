"""Result classes for stationarity analysis."""

from dataclasses import dataclass
from typing import Optional, Dict, Callable, Any
import pandas as pd
import numpy as np


@dataclass
class TestResult:
    """
    Results from a stationarity test.
    
    Attributes:
        test_name: Name of the test performed
        test_statistic: The test statistic value
        p_value: P-value from the test
        critical_values: Dictionary of critical values at different significance levels
        is_stationary: Whether the series is stationary according to this test
        alpha: Significance level used
        interpretation: Human-readable interpretation of the result
    """
    
    test_name: str
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    is_stationary: bool
    alpha: float
    interpretation: str
    
    def __str__(self) -> str:
        """String representation of test results."""
        result = f"\n{self.test_name} Results:\n"
        result += f"{'='*50}\n"
        result += f"Test Statistic: {self.test_statistic:.4f}\n"
        result += f"P-value: {self.p_value:.4f}\n"
        result += f"Is Stationary: {self.is_stationary}\n"
        result += f"\nCritical Values:\n"
        for level, value in self.critical_values.items():
            result += f"  {level}: {value:.4f}\n"
        result += f"\nInterpretation:\n{self.interpretation}\n"
        return result


@dataclass
class TransformationResult:
    """
    Results from applying a transformation.
    
    Attributes:
        transformation_name: Name of the transformation applied
        transformation_params: Parameters used in the transformation
        original_data: Original time series data
        transformed_data: Transformed time series data
        inverse_function: Function to reverse the transformation
        improvement_metric: Metric showing improvement (e.g., p-value change)
    """
    
    transformation_name: str
    transformation_params: Dict[str, Any]
    original_data: pd.Series
    transformed_data: pd.Series
    inverse_function: Optional[Callable]
    improvement_metric: float
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply inverse transformation to data.
        
        Args:
            data: Transformed data to reverse
            
        Returns:
            Data in original scale
        """
        if self.inverse_function is None:
            return data
        return self.inverse_function(data)
    
    def __str__(self) -> str:
        """String representation of transformation results."""
        result = f"\nTransformation: {self.transformation_name}\n"
        result += f"{'='*50}\n"
        result += f"Parameters: {self.transformation_params}\n"
        result += f"Improvement Metric: {self.improvement_metric:.4f}\n"
        return result


@dataclass
class StationarityResult:
    """
    Comprehensive results from stationarity analysis.
    
    Attributes:
        is_stationary: Overall stationarity status
        trend_stationary: Whether trend is stationary
        variance_stationary: Whether variance is stationary
        trend_test_result: Results from trend stationarity test
        variance_test_result: Results from variance stationarity test
        trend_transformation: Transformation applied for trend (if any)
        variance_transformation: Transformation applied for variance (if any)
        final_data: Final transformed data
        original_data: Original input data
        recommendations: List of recommendations for the user
    """
    
    is_stationary: bool
    trend_stationary: bool
    variance_stationary: bool
    trend_test_result: Optional[TestResult]
    variance_test_result: Optional[TestResult]
    trend_transformation: Optional[TransformationResult]
    variance_transformation: Optional[TransformationResult]
    final_data: pd.Series
    original_data: pd.Series
    recommendations: list
    
    def get_inverse_transform(self) -> Callable:
        """
        Get a function that applies all inverse transformations in correct order.
        
        Returns:
            Function that transforms data back to original scale
        """
        def inverse(data: np.ndarray) -> np.ndarray:
            """Apply all inverse transformations."""
            result = data.copy()
            
            # Apply trend inverse first (if exists)
            if self.trend_transformation is not None:
                result = self.trend_transformation.inverse_transform(result)
            
            # Then apply variance inverse (if exists)
            if self.variance_transformation is not None:
                result = self.variance_transformation.inverse_transform(result)
            
            return result
        
        return inverse
    
    def summary(self) -> str:
        """
        Generate a comprehensive summary of the analysis.
        
        Returns:
            Formatted string with all results
        """
        summary = "\n" + "="*70 + "\n"
        summary += "STATIONARITY ANALYSIS SUMMARY\n"
        summary += "="*70 + "\n\n"
        
        summary += f"Overall Stationary: {self.is_stationary}\n"
        summary += f"Trend Stationary: {self.trend_stationary}\n"
        summary += f"Variance Stationary: {self.variance_stationary}\n\n"
        
        if self.trend_test_result:
            summary += str(self.trend_test_result)
        
        if self.variance_test_result:
            summary += str(self.variance_test_result)
        
        if self.trend_transformation:
            summary += str(self.trend_transformation)
        
        if self.variance_transformation:
            summary += str(self.variance_transformation)
        
        if self.recommendations:
            summary += "\nRECOMMENDATIONS:\n"
            summary += "-"*70 + "\n"
            for i, rec in enumerate(self.recommendations, 1):
                summary += f"{i}. {rec}\n"
        
        summary += "\n" + "="*70 + "\n"
        return summary
    
    def __str__(self) -> str:
        """String representation."""
        return self.summary()
