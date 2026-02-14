"""
Integration tests for the complete StationarityToolkit.

Tests the full pipeline from testing to transformation.
"""

import pytest
import numpy as np
import pandas as pd
from src.stationarity_toolkit_v2 import StationarityToolkit, StationarityConfig


class TestBasicUsage:
    """Tests for basic toolkit usage."""
    
    def test_initialization_default(self):
        """Test toolkit initialization with defaults."""
        toolkit = StationarityToolkit(alpha=0.05)
        
        assert toolkit.config.alpha == 0.05
        assert toolkit.config.variance_test == "levene"
        assert toolkit.config.trend_test == "adf"
    
    def test_initialization_with_config(self):
        """Test toolkit initialization with custom config."""
        config = StationarityConfig(
            alpha=0.01,
            variance_test="bartlett",
            trend_test="kpss",
            verbose=True
        )
        toolkit = StationarityToolkit(config=config)
        
        assert toolkit.config.alpha == 0.01
        assert toolkit.config.variance_test == "bartlett"
        assert toolkit.config.trend_test == "kpss"
        assert toolkit.config.verbose is True


class TestStationarityTesting:
    """Tests for test_stationarity method."""
    
    def test_stationary_series(self):
        """Test with stationary series."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        toolkit = StationarityToolkit(alpha=0.05)
        result = toolkit.test_stationarity(data)
        
        assert result.is_stationary == True
        assert result.trend_stationary == True
        assert result.variance_stationary == True
        assert result.trend_test_result is not None
        assert result.variance_test_result is not None
    
    def test_nonstationary_trend(self):
        """Test with series having trend."""
        np.random.seed(42)
        n = 200
        data = pd.Series(np.cumsum(np.random.normal(0.1, 1, n)))
        
        toolkit = StationarityToolkit(alpha=0.05)
        result = toolkit.test_stationarity(data)
        
        assert result.is_stationary == False
        assert result.trend_stationary == False
    
    def test_nonstationary_variance(self):
        """Test with series having changing variance."""
        np.random.seed(42)
        n = 200
        data = pd.Series(np.random.normal(0, 1 + np.linspace(0, 2, n), n))
        
        toolkit = StationarityToolkit(alpha=0.05)
        result = toolkit.test_stationarity(data)
        
        assert result.variance_stationary == False
    
    def test_test_variance_only(self):
        """Test with only variance testing."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        toolkit = StationarityToolkit(alpha=0.05)
        result = toolkit.test_stationarity(data, test_variance=True, test_trend=False)
        
        assert result.variance_test_result is not None
        assert result.trend_test_result is None
    
    def test_test_trend_only(self):
        """Test with only trend testing."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        toolkit = StationarityToolkit(alpha=0.05)
        result = toolkit.test_stationarity(data, test_variance=False, test_trend=True)
        
        assert result.trend_test_result is not None
        assert result.variance_test_result is None


class TestMakeStationary:
    """Tests for make_stationary method."""
    
    def test_already_stationary(self):
        """Test with already stationary series."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        toolkit = StationarityToolkit(alpha=0.05)
        result = toolkit.make_stationary(data)
        
        assert result.is_stationary == True
        assert result.variance_transformation is None
        assert result.trend_transformation is None
    
    def test_variance_nonstationarity(self):
        """Test handling variance non-stationarity."""
        np.random.seed(42)
        n = 200
        # Data with clearly changing variance
        data = pd.Series(np.random.normal(0, 1 + 2 * np.linspace(0, 1, n), n))
        
        toolkit = StationarityToolkit(alpha=0.05)
        result = toolkit.make_stationary(data, handle_variance=True, handle_trend=False)
        
        # Should apply variance transformation
        assert result.variance_transformation is not None
        assert result.variance_transformation.transformation_name in [
            "Log Transform", "Square Root Transform", "Box-Cox Transform"
        ]
    
    def test_trend_nonstationarity(self):
        """Test handling trend non-stationarity."""
        np.random.seed(42)
        n = 200
        # Random walk
        data = pd.Series(np.cumsum(np.random.normal(0, 1, n)))
        
        toolkit = StationarityToolkit(alpha=0.05)
        result = toolkit.make_stationary(data, handle_variance=False, handle_trend=True)
        
        # Should apply trend transformation
        assert result.trend_transformation is not None
        assert "Differencing" in result.trend_transformation.transformation_name
    
    def test_both_nonstationarities(self):
        """Test handling both variance and trend non-stationarity."""
        np.random.seed(42)
        n = 200
        # Exponential trend with changing variance
        t = np.arange(n)
        data = pd.Series(np.exp(t * 0.01) * (1 + np.random.normal(0, 0.1, n)))
        
        toolkit = StationarityToolkit(alpha=0.05)
        result = toolkit.make_stationary(data)
        
        # May apply both transformations
        assert result.final_data is not None
        assert len(result.final_data) > 0
    
    def test_seasonal_data(self):
        """Test with seasonal data."""
        np.random.seed(42)
        n = 104  # 2 years of weekly data
        t = np.arange(n)
        # Trend + seasonality
        trend = t * 0.1
        seasonal = 5 * np.sin(2 * np.pi * t / 52)
        noise = np.random.normal(0, 1, n)
        data = pd.Series(trend + seasonal + noise)
        
        config = StationarityConfig(
            alpha=0.05,
            seasonal_period=52,
            auto_detect_seasonality=False
        )
        toolkit = StationarityToolkit(config=config)
        result = toolkit.make_stationary(data, seasonal_period=52)
        
        assert result.trend_transformation is not None
    
    def test_auto_seasonality_detection(self):
        """Test automatic seasonality detection."""
        np.random.seed(42)
        n = 104
        t = np.arange(n)
        # Strong seasonal pattern
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        noise = np.random.normal(0, 1, n)
        data = pd.Series(seasonal + noise)
        
        config = StationarityConfig(
            alpha=0.05,
            auto_detect_seasonality=True
        )
        toolkit = StationarityToolkit(config=config)
        result = toolkit.make_stationary(data)
        
        # Should work without errors
        assert result is not None
    
    def test_min_observations_check(self):
        """Test that minimum observations requirement is enforced."""
        data = pd.Series(np.random.normal(0, 1, 30))
        
        config = StationarityConfig(min_observations=50)
        toolkit = StationarityToolkit(config=config)
        
        with pytest.raises(ValueError, match="Time series too short"):
            toolkit.make_stationary(data)


class TestInverseTransformations:
    """Tests for inverse transformations."""
    
    def test_variance_inverse(self):
        """Test inverse of variance transformation."""
        np.random.seed(42)
        data = pd.Series(np.exp(np.random.normal(0, 0.5, 100)))
        
        toolkit = StationarityToolkit(alpha=0.05)
        result = toolkit.make_stationary(data, handle_variance=True, handle_trend=False)
        
        if result.variance_transformation is not None:
            # Get inverse function
            inverse_func = result.get_inverse_transform()
            
            # Apply to transformed data
            reconstructed = inverse_func(result.final_data.values)
            
            # Should be close to original (accounting for any dropped values)
            min_len = min(len(reconstructed), len(data))
            np.testing.assert_array_almost_equal(
                reconstructed[:min_len],
                data.values[:min_len],
                decimal=6
            )
    
    def test_trend_inverse(self):
        """Test inverse of trend transformation."""
        np.random.seed(42)
        data = pd.Series(np.cumsum(np.random.normal(0, 1, 100)))
        
        toolkit = StationarityToolkit(alpha=0.05)
        result = toolkit.make_stationary(data, handle_variance=False, handle_trend=True)
        
        if result.trend_transformation is not None:
            # Get inverse function
            inverse_func = result.get_inverse_transform()
            
            # Apply to transformed data
            reconstructed = inverse_func(result.final_data.values)
            
            # Should be close to original
            np.testing.assert_array_almost_equal(
                reconstructed,
                data.values,
                decimal=6
            )
    
    def test_combined_inverse(self):
        """Test inverse of combined transformations."""
        np.random.seed(42)
        n = 100
        # Exponential trend
        data = pd.Series(np.exp(np.arange(n) * 0.01 + np.random.normal(0, 0.1, n)))
        
        toolkit = StationarityToolkit(alpha=0.05)
        result = toolkit.make_stationary(data)
        
        # Get combined inverse
        inverse_func = result.get_inverse_transform()
        
        # Should work without errors
        reconstructed = inverse_func(result.final_data.values)
        assert len(reconstructed) > 0


class TestResultSummary:
    """Tests for result summary and reporting."""
    
    def test_summary_generation(self):
        """Test that summary is generated correctly."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        toolkit = StationarityToolkit(alpha=0.05)
        result = toolkit.make_stationary(data)
        
        summary = result.summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "STATIONARITY ANALYSIS SUMMARY" in summary
        assert "Overall Stationary" in summary
    
    def test_recommendations_generated(self):
        """Test that recommendations are generated."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        toolkit = StationarityToolkit(alpha=0.05)
        result = toolkit.make_stationary(data)
        
        assert result.recommendations is not None
        assert len(result.recommendations) > 0
        assert isinstance(result.recommendations[0], str)


class TestDifferentConfigurations:
    """Tests with different configuration options."""
    
    def test_different_variance_tests(self):
        """Test with different variance tests."""
        np.random.seed(42)
        n = 200
        data = pd.Series(np.random.normal(0, 1 + np.linspace(0, 1, n), n))
        
        for test in ["levene", "bartlett", "white", "arch"]:
            config = StationarityConfig(variance_test=test)
            toolkit = StationarityToolkit(config=config)
            
            result = toolkit.test_stationarity(data)
            assert result.variance_test_result is not None
    
    def test_different_trend_tests(self):
        """Test with different trend tests."""
        np.random.seed(42)
        data = pd.Series(np.cumsum(np.random.normal(0, 1, 200)))
        
        for test in ["adf", "kpss", "pp"]:
            config = StationarityConfig(trend_test=test)
            toolkit = StationarityToolkit(config=config)
            
            result = toolkit.test_stationarity(data)
            assert result.trend_test_result is not None
    
    def test_different_alpha_levels(self):
        """Test with different significance levels."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        for alpha in [0.01, 0.05, 0.10]:
            toolkit = StationarityToolkit(alpha=alpha)
            result = toolkit.test_stationarity(data)
            
            assert result.trend_test_result.alpha == alpha
            assert result.variance_test_result.alpha == alpha


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_constant_series(self):
        """Test with constant series."""
        data = pd.Series([5.0] * 100)
        
        toolkit = StationarityToolkit(alpha=0.05)
        
        # Should handle gracefully (may fail some tests)
        try:
            result = toolkit.test_stationarity(data)
            assert result is not None
        except Exception as e:
            # Some tests may fail with constant data, that's okay
            assert "constant" in str(e).lower() or "variance" in str(e).lower()
    
    def test_series_with_nans(self):
        """Test with series containing NaN values."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        data.iloc[50:60] = np.nan
        
        toolkit = StationarityToolkit(alpha=0.05)
        result = toolkit.test_stationarity(data)
        
        # Should handle NaNs by dropping them
        assert result is not None
    
    def test_very_short_series(self):
        """Test with very short series."""
        data = pd.Series([1, 2, 3, 4, 5])
        
        toolkit = StationarityToolkit(alpha=0.05)
        
        # Should raise error or handle gracefully
        with pytest.raises((ValueError, Exception)):
            toolkit.make_stationary(data)


class TestComparisonWithOldToolkit:
    """
    Tests comparing v2.0 with the old toolkit behavior.
    
    Demonstrates the improvements and fixes.
    """
    
    def test_variance_test_correctness(self):
        """
        Demonstrate that v2.0 uses correct variance tests.
        
        Old toolkit used Phillips-Perron for "variance" (WRONG).
        New toolkit uses Levene/Bartlett/White/ARCH (CORRECT).
        """
        np.random.seed(42)
        n = 200
        # Series with changing variance but no trend
        data = pd.Series(np.random.normal(0, 1 + np.linspace(0, 2, n), n))
        
        # New toolkit (v2.0) - correct approach
        toolkit_v2 = StationarityToolkit(alpha=0.05)
        result_v2 = toolkit_v2.test_stationarity(data)
        
        # Should correctly identify variance non-stationarity
        assert result_v2.variance_stationary == False, \
            "v2.0 should detect variance non-stationarity"
        
        # Old toolkit would have used Phillips-Perron for "variance"
        from arch.unitroot import PhillipsPerron
        pp_test = PhillipsPerron(data.dropna())
        pp_says_stationary = pp_test.pvalue < 0.05
        
        print(f"\nv2.0 Levene test: variance {'stationary' if result_v2.variance_stationary else 'non-stationary'}")
        print(f"Old PP test (WRONG for variance): {'stationary' if pp_says_stationary else 'non-stationary'}")
        print("v2.0 correctly identifies the variance issue!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
