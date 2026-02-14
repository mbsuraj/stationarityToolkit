"""
Unit tests for variance and trend transformations.

Tests that transformations work correctly and inverse functions restore original data.
"""

import pytest
import numpy as np
import pandas as pd
from src.stationarity_toolkit_v2.transformations.variance_transformations import (
    log_transform,
    sqrt_transform,
    boxcox_transform,
    apply_variance_transformation
)
from src.stationarity_toolkit_v2.transformations.trend_transformations import (
    difference,
    seasonal_difference,
    combined_difference,
    apply_trend_transformation
)


class TestLogTransform:
    """Tests for logarithmic transformation."""
    
    def test_positive_data(self):
        """Test with positive data."""
        np.random.seed(42)
        data = pd.Series(np.abs(np.random.normal(5, 2, 100)))
        
        result = log_transform(data)
        
        assert result.transformation_name == "Log Transform"
        assert len(result.transformed_data) == len(data)
        assert result.inverse_function is not None
    
    def test_inverse_transformation(self):
        """Test that inverse transformation recovers original data."""
        np.random.seed(42)
        data = pd.Series(np.abs(np.random.normal(5, 2, 100)))
        
        result = log_transform(data)
        reconstructed = result.inverse_function(result.transformed_data.values)
        
        # Should be very close to original
        np.testing.assert_array_almost_equal(reconstructed, data.values, decimal=10)
    
    def test_negative_data_with_offset(self):
        """Test that negative data is handled with offset."""
        data = pd.Series([-5, -3, -1, 1, 3, 5])
        
        result = log_transform(data)
        
        # Should have added offset
        assert result.transformation_params['offset'] > 0
        assert not np.any(np.isnan(result.transformed_data))
    
    def test_variance_reduction(self):
        """Test that log transform reduces variance for exponential-like data."""
        np.random.seed(42)
        # Data with variance proportional to level
        data = pd.Series(np.exp(np.random.normal(0, 0.5, 100)))
        
        result = log_transform(data)
        
        # Transformed data should have lower variance
        assert result.transformed_data.var() < data.var()


class TestSqrtTransform:
    """Tests for square root transformation."""
    
    def test_positive_data(self):
        """Test with positive data."""
        np.random.seed(42)
        data = pd.Series(np.abs(np.random.normal(5, 2, 100)))
        
        result = sqrt_transform(data)
        
        assert result.transformation_name == "Square Root Transform"
        assert len(result.transformed_data) == len(data)
    
    def test_inverse_transformation(self):
        """Test that inverse transformation recovers original data."""
        np.random.seed(42)
        data = pd.Series(np.abs(np.random.normal(5, 2, 100)))
        
        result = sqrt_transform(data)
        reconstructed = result.inverse_function(result.transformed_data.values)
        
        np.testing.assert_array_almost_equal(reconstructed, data.values, decimal=10)
    
    def test_count_data(self):
        """Test with count data (Poisson-like)."""
        np.random.seed(42)
        data = pd.Series(np.random.poisson(10, 100).astype(float))
        
        result = sqrt_transform(data)
        
        # Should work without issues
        assert not np.any(np.isnan(result.transformed_data))


class TestBoxCoxTransform:
    """Tests for Box-Cox transformation."""
    
    def test_positive_data(self):
        """Test with positive data."""
        np.random.seed(42)
        data = pd.Series(np.abs(np.random.normal(5, 2, 100)))
        
        result = boxcox_transform(data)
        
        assert result.transformation_name == "Box-Cox Transform"
        assert 'lambda' in result.transformation_params
        assert result.transformation_params['lambda'] is not None
    
    def test_inverse_transformation(self):
        """Test that inverse transformation recovers original data."""
        np.random.seed(42)
        data = pd.Series(np.abs(np.random.normal(5, 2, 100)))
        
        result = boxcox_transform(data)
        reconstructed = result.inverse_function(result.transformed_data.values)
        
        np.testing.assert_array_almost_equal(reconstructed, data.values, decimal=8)
    
    def test_lambda_optimization(self):
        """Test that lambda is optimized."""
        np.random.seed(42)
        data = pd.Series(np.abs(np.random.normal(5, 2, 100)))
        
        result = boxcox_transform(data)
        
        # Lambda should be optimized (not None)
        assert result.transformation_params['lambda'] is not None
        # Lambda should be reasonable
        assert -5 < result.transformation_params['lambda'] < 5
    
    def test_fixed_lambda(self):
        """Test with fixed lambda parameter."""
        np.random.seed(42)
        data = pd.Series(np.abs(np.random.normal(5, 2, 100)))
        
        result = boxcox_transform(data, lmbda=0.5)
        
        assert result.transformation_params['lambda'] == 0.5


class TestApplyVarianceTransformation:
    """Tests for automatic variance transformation selection."""
    
    def test_log_method(self):
        """Test explicit log method."""
        np.random.seed(42)
        data = pd.Series(np.abs(np.random.normal(5, 2, 100)))
        
        result = apply_variance_transformation(data, method="log")
        
        assert "Log" in result.transformation_name
    
    def test_sqrt_method(self):
        """Test explicit sqrt method."""
        np.random.seed(42)
        data = pd.Series(np.abs(np.random.normal(5, 2, 100)))
        
        result = apply_variance_transformation(data, method="sqrt")
        
        assert "Square Root" in result.transformation_name
    
    def test_boxcox_method(self):
        """Test explicit boxcox method."""
        np.random.seed(42)
        data = pd.Series(np.abs(np.random.normal(5, 2, 100)))
        
        result = apply_variance_transformation(data, method="boxcox")
        
        assert "Box-Cox" in result.transformation_name
    
    def test_auto_method(self):
        """Test automatic method selection."""
        np.random.seed(42)
        data = pd.Series(np.abs(np.random.normal(5, 2, 100)))
        
        result = apply_variance_transformation(data, method="auto")
        
        # Should select one of the methods
        assert result.transformation_name in [
            "Log Transform", "Square Root Transform", "Box-Cox Transform"
        ]
    
    def test_none_method(self):
        """Test no transformation."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 100))
        
        result = apply_variance_transformation(data, method="none")
        
        assert result.transformation_name == "None"
        np.testing.assert_array_equal(result.transformed_data.values, data.values)
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        data = pd.Series(np.random.normal(0, 1, 100))
        
        with pytest.raises(ValueError, match="Unknown variance transformation"):
            apply_variance_transformation(data, method="invalid")


class TestDifference:
    """Tests for differencing transformation."""
    
    def test_first_difference(self):
        """Test first-order differencing."""
        data = pd.Series([1, 3, 6, 10, 15, 21])
        
        result = difference(data, periods=1, order=1)
        
        assert result.transformation_name == "Differencing (order=1, periods=1)"
        # First differences should be [2, 3, 4, 5, 6]
        expected = [2, 3, 4, 5, 6]
        np.testing.assert_array_almost_equal(
            result.transformed_data.values, expected, decimal=10
        )
    
    def test_second_difference(self):
        """Test second-order differencing."""
        data = pd.Series([1, 3, 6, 10, 15, 21])
        
        result = difference(data, periods=1, order=2)
        
        # Second differences should be constant [1, 1, 1, 1]
        assert len(result.transformed_data) == 4
    
    def test_inverse_first_difference(self):
        """Test inverse of first differencing."""
        data = pd.Series([1.0, 3.0, 6.0, 10.0, 15.0, 21.0])
        
        result = difference(data, periods=1, order=1)
        reconstructed = result.inverse_function(result.transformed_data.values)
        
        # Should recover original (accounting for lost first value)
        np.testing.assert_array_almost_equal(
            reconstructed, data.values, decimal=10
        )
    
    def test_removes_linear_trend(self):
        """Test that differencing removes linear trend."""
        np.random.seed(42)
        n = 100
        # Linear trend + noise
        data = pd.Series(np.arange(n) * 0.5 + np.random.normal(0, 0.1, n))
        
        result = difference(data, periods=1, order=1)
        
        # Differenced data should have much lower variance
        assert result.transformed_data.var() < data.var()


class TestSeasonalDifference:
    """Tests for seasonal differencing."""
    
    def test_seasonal_difference(self):
        """Test seasonal differencing."""
        # Create data with period 4
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        
        result = seasonal_difference(data, seasonal_period=4, order=1)
        
        assert "Seasonal Differencing" in result.transformation_name
        assert result.transformation_params['seasonal_period'] == 4
        # Should have 12 - 4 = 8 values
        assert len(result.transformed_data) == 8
    
    def test_removes_seasonality(self):
        """Test that seasonal differencing removes seasonal pattern."""
        np.random.seed(42)
        n = 104  # 2 years of weekly data
        # Create seasonal pattern
        t = np.arange(n)
        seasonal = 10 * np.sin(2 * np.pi * t / 52)
        noise = np.random.normal(0, 1, n)
        data = pd.Series(seasonal + noise)
        
        result = seasonal_difference(data, seasonal_period=52, order=1)
        
        # Seasonally differenced data should have lower variance
        assert result.transformed_data.var() < data.var()
    
    def test_invalid_period(self):
        """Test that invalid seasonal period raises error."""
        data = pd.Series(np.arange(10))
        
        with pytest.raises(ValueError, match="seasonal_period must be >= 2"):
            seasonal_difference(data, seasonal_period=1)


class TestCombinedDifference:
    """Tests for combined differencing."""
    
    def test_trend_and_seasonal(self):
        """Test combined trend and seasonal differencing."""
        np.random.seed(42)
        n = 104
        t = np.arange(n)
        # Trend + seasonality + noise
        trend = t * 0.1
        seasonal = 5 * np.sin(2 * np.pi * t / 52)
        noise = np.random.normal(0, 0.5, n)
        data = pd.Series(trend + seasonal + noise)
        
        result = combined_difference(
            data,
            seasonal_period=52,
            trend_order=1,
            seasonal_order=1
        )
        
        assert "Combined Differencing" in result.transformation_name
        # Should remove both trend and seasonality
        assert result.transformed_data.var() < data.var()
    
    def test_trend_only(self):
        """Test combined with only trend differencing."""
        data = pd.Series(np.arange(100) * 0.5)
        
        result = combined_difference(
            data,
            seasonal_period=None,
            trend_order=1,
            seasonal_order=0
        )
        
        assert result.transformation_params['trend_order'] == 1
        assert result.transformation_params['seasonal_order'] == 0


class TestApplyTrendTransformation:
    """Tests for automatic trend transformation selection."""
    
    def test_difference_method(self):
        """Test explicit difference method."""
        data = pd.Series(np.arange(100) * 0.5)
        
        result = apply_trend_transformation(data, method="difference")
        
        assert "Differencing" in result.transformation_name
    
    def test_seasonal_method(self):
        """Test explicit seasonal method."""
        data = pd.Series(np.arange(100))
        
        result = apply_trend_transformation(
            data, method="seasonal", seasonal_period=12
        )
        
        assert "Seasonal" in result.transformation_name
    
    def test_seasonal_requires_period(self):
        """Test that seasonal method requires period."""
        data = pd.Series(np.arange(100))
        
        with pytest.raises(ValueError, match="seasonal_period required"):
            apply_trend_transformation(data, method="seasonal")
    
    def test_combined_method(self):
        """Test explicit combined method."""
        data = pd.Series(np.arange(100))
        
        result = apply_trend_transformation(
            data, method="combined", seasonal_period=12
        )
        
        assert "Combined" in result.transformation_name
    
    def test_auto_method(self):
        """Test automatic method selection."""
        np.random.seed(42)
        data = pd.Series(np.cumsum(np.random.normal(0, 1, 100)))
        
        result = apply_trend_transformation(data, method="auto")
        
        # Should select some differencing method
        assert "Differencing" in result.transformation_name or "None" in result.transformation_name
    
    def test_none_method(self):
        """Test no transformation."""
        data = pd.Series(np.random.normal(0, 1, 100))
        
        result = apply_trend_transformation(data, method="none")
        
        assert result.transformation_name == "None"
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        data = pd.Series(np.arange(100))
        
        with pytest.raises(ValueError, match="Unknown trend transformation"):
            apply_trend_transformation(data, method="invalid")


class TestTransformationInverses:
    """
    Comprehensive tests for transformation inverses.
    
    Critical for forecasting - we need to transform predictions back to original scale.
    """
    
    def test_all_variance_transforms_invertible(self):
        """Test that all variance transforms are invertible."""
        np.random.seed(42)
        data = pd.Series(np.abs(np.random.normal(5, 2, 100)))
        
        transforms = [
            log_transform(data),
            sqrt_transform(data),
            boxcox_transform(data)
        ]
        
        for result in transforms:
            reconstructed = result.inverse_function(result.transformed_data.values)
            np.testing.assert_array_almost_equal(
                reconstructed, data.values, decimal=8,
                err_msg=f"{result.transformation_name} inverse failed"
            )
    
    def test_all_trend_transforms_invertible(self):
        """Test that all trend transforms are invertible."""
        np.random.seed(42)
        data = pd.Series(np.cumsum(np.random.normal(0, 1, 100)))
        
        transforms = [
            difference(data, periods=1, order=1),
            difference(data, periods=1, order=2),
        ]
        
        for result in transforms:
            reconstructed = result.inverse_function(result.transformed_data.values)
            np.testing.assert_array_almost_equal(
                reconstructed, data.values, decimal=8,
                err_msg=f"{result.transformation_name} inverse failed"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
