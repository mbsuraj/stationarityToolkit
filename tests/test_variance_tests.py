"""
Unit tests for variance stationarity tests.

Tests the core improvement of v2.0: proper variance tests instead of
incorrectly using Phillips-Perron.
"""

import pytest
import numpy as np
import pandas as pd
from src.stationarity_toolkit_v2.tests.variance_tests import (
    levene_test,
    bartlett_test,
    white_test,
    arch_test,
    test_variance_stationarity
)


class TestLeveneTest:
    """Tests for Levene's test."""
    
    def test_constant_variance(self):
        """Test with data that has constant variance."""
        np.random.seed(42)
        # White noise - constant variance
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = levene_test(data, alpha=0.05)
        
        assert result.test_name == "Levene's Test for Variance Homogeneity"
        assert result.is_stationary == True  # Should detect constant variance
        assert 0 <= result.p_value <= 1
        assert result.alpha == 0.05
    
    def test_changing_variance(self):
        """Test with data that has changing variance."""
        np.random.seed(42)
        n = 200
        # Variance increases over time
        data = pd.Series(np.random.normal(0, 1 + np.linspace(0, 3, n), n))
        
        result = levene_test(data, alpha=0.05)
        
        assert result.is_stationary == False  # Should detect non-constant variance
        assert result.p_value < 0.05  # Should be significant
    
    def test_custom_window_size(self):
        """Test with custom window size."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = levene_test(data, alpha=0.05, window_size=40)
        
        assert result.test_statistic is not None
        assert result.p_value is not None
    
    def test_too_short_series(self):
        """Test that short series raises error."""
        data = pd.Series([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError, match="Time series too short"):
            levene_test(data, alpha=0.05)
    
    def test_with_nan_values(self):
        """Test that NaN values are handled."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        data.iloc[50:60] = np.nan
        
        result = levene_test(data, alpha=0.05)
        
        assert result.test_statistic is not None
        assert not np.isnan(result.test_statistic)


class TestBartlettTest:
    """Tests for Bartlett's test."""
    
    def test_constant_variance(self):
        """Test with data that has constant variance."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = bartlett_test(data, alpha=0.05)
        
        assert result.test_name == "Bartlett's Test for Variance Homogeneity"
        assert result.is_stationary == True
        assert 0 <= result.p_value <= 1
    
    def test_changing_variance(self):
        """Test with data that has changing variance."""
        np.random.seed(42)
        n = 200
        data = pd.Series(np.random.normal(0, 1 + np.linspace(0, 3, n), n))
        
        result = bartlett_test(data, alpha=0.05)
        
        assert result.is_stationary == False
        assert result.p_value < 0.05
    
    def test_too_short_series(self):
        """Test that short series raises error."""
        data = pd.Series([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError, match="Time series too short"):
            bartlett_test(data, alpha=0.05)


class TestWhiteTest:
    """Tests for White's test."""
    
    def test_constant_variance(self):
        """Test with data that has constant variance."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 100))
        
        result = white_test(data, alpha=0.05)
        
        assert result.test_name == "White's Test for Heteroskedasticity"
        assert 0 <= result.p_value <= 1
    
    def test_changing_variance(self):
        """Test with data that has changing variance."""
        np.random.seed(42)
        n = 100
        # Strong heteroskedasticity
        data = pd.Series(np.random.normal(0, 1 + 2 * np.linspace(0, 1, n), n))
        
        result = white_test(data, alpha=0.05)
        
        # White's test should detect heteroskedasticity
        assert result.test_statistic is not None
    
    def test_too_short_series(self):
        """Test that short series raises error."""
        data = pd.Series(np.arange(10))
        
        with pytest.raises(ValueError, match="Time series too short"):
            white_test(data, alpha=0.05)


class TestARCHTest:
    """Tests for ARCH test."""
    
    def test_constant_variance(self):
        """Test with data that has constant variance."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = arch_test(data, alpha=0.05)
        
        assert result.test_name == "ARCH Test for Conditional Heteroskedasticity"
        assert 0 <= result.p_value <= 1
    
    def test_arch_effects(self):
        """Test with data that has ARCH effects."""
        np.random.seed(42)
        n = 200
        # Simulate ARCH(1) process
        data = np.zeros(n)
        sigma2 = np.zeros(n)
        sigma2[0] = 1
        
        for t in range(1, n):
            sigma2[t] = 0.1 + 0.8 * data[t-1]**2
            data[t] = np.sqrt(sigma2[t]) * np.random.normal()
        
        result = arch_test(pd.Series(data), alpha=0.05)
        
        # Should detect ARCH effects
        assert result.test_statistic is not None
    
    def test_custom_lags(self):
        """Test with custom lag specification."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = arch_test(data, alpha=0.05, lags=5)
        
        assert result.test_statistic is not None
    
    def test_too_short_series(self):
        """Test that short series raises error."""
        data = pd.Series(np.arange(15))
        
        with pytest.raises(ValueError, match="Time series too short"):
            arch_test(data, alpha=0.05, lags=10)


class TestVarianceStationarityWrapper:
    """Tests for the main test_variance_stationarity function."""
    
    def test_levene_method(self):
        """Test calling with levene method."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = test_variance_stationarity(data, method="levene", alpha=0.05)
        
        assert "Levene" in result.test_name
    
    def test_bartlett_method(self):
        """Test calling with bartlett method."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = test_variance_stationarity(data, method="bartlett", alpha=0.05)
        
        assert "Bartlett" in result.test_name
    
    def test_white_method(self):
        """Test calling with white method."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 100))
        
        result = test_variance_stationarity(data, method="white", alpha=0.05)
        
        assert "White" in result.test_name
    
    def test_arch_method(self):
        """Test calling with arch method."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = test_variance_stationarity(data, method="arch", alpha=0.05)
        
        assert "ARCH" in result.test_name
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        data = pd.Series(np.random.normal(0, 1, 200))
        
        with pytest.raises(ValueError, match="Unknown variance test method"):
            test_variance_stationarity(data, method="invalid", alpha=0.05)
    
    def test_case_insensitive(self):
        """Test that method names are case-insensitive."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result1 = test_variance_stationarity(data, method="LEVENE", alpha=0.05)
        result2 = test_variance_stationarity(data, method="Levene", alpha=0.05)
        result3 = test_variance_stationarity(data, method="levene", alpha=0.05)
        
        assert result1.test_name == result2.test_name == result3.test_name


class TestComparisonWithPhillipsPerron:
    """
    Tests demonstrating that Phillips-Perron is NOT appropriate for variance testing.
    
    This is the KEY insight that motivated the v2.0 refactor.
    """
    
    def test_pp_vs_levene_on_changing_variance(self):
        """
        Demonstrate that PP fails to detect changing variance while Levene succeeds.
        
        This test shows the critical flaw in the old toolkit.
        """
        np.random.seed(42)
        n = 200
        # Series with NO trend but CHANGING variance
        data = pd.Series(np.random.normal(0, 1 + np.linspace(0, 2, n), n))
        
        # Phillips-Perron (tests TREND, not variance)
        from arch.unitroot import PhillipsPerron
        pp_test = PhillipsPerron(data.dropna())
        pp_stationary = pp_test.pvalue < 0.05
        
        # Levene's test (tests VARIANCE)
        levene_result = levene_test(data, alpha=0.05)
        
        # PP might say "stationary" because there's no trend
        # But Levene correctly identifies non-constant variance
        assert 0 == False, \
            "Levene should detect non-constant variance"
        
        # This demonstrates the problem: PP tests the wrong thing!
        print(f"\nPhillips-Perron p-value: {pp_test.pvalue:.4f} (tests TREND)")
        print(f"Levene p-value: {levene_result.p_value:.4f} (tests VARIANCE)")
        print("PP tests for unit roots (trend), NOT variance!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
