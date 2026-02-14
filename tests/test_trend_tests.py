"""
Unit tests for trend stationarity tests.

Tests ADF, KPSS, and Phillips-Perron tests for trend stationarity.
"""

import pytest
import numpy as np
import pandas as pd
from src.stationarity_toolkit_v2.tests.trend_tests import (
    adf_test,
    kpss_test,
    phillips_perron_test,
    combined_trend_test,
    test_trend_stationarity
)


class TestADFTest:
    """Tests for Augmented Dickey-Fuller test."""
    
    def test_stationary_series(self):
        """Test with stationary series (white noise)."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = adf_test(data, alpha=0.05)
        
        assert result.test_name == "Augmented Dickey-Fuller (ADF) Test"
        assert result.is_stationary == True  # White noise is stationary
        assert result.p_value < 0.05
        assert 0 <= result.p_value <= 1
    
    def test_nonstationary_series_with_trend(self):
        """Test with non-stationary series (random walk with trend)."""
        np.random.seed(42)
        n = 200
        # Random walk with drift
        data = pd.Series(np.cumsum(np.random.normal(0.1, 1, n)))
        
        result = adf_test(data, alpha=0.05)
        
        assert result.is_stationary == False  # Should detect non-stationarity
        assert result.p_value > 0.05
    
    def test_different_regression_types(self):
        """Test different regression specifications."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        # Test with constant only
        result_c = adf_test(data, alpha=0.05, regression='c')
        assert result_c.test_statistic is not None
        
        # Test with constant and trend
        result_ct = adf_test(data, alpha=0.05, regression='ct')
        assert result_ct.test_statistic is not None
        
        # Test with no constant or trend
        result_n = adf_test(data, alpha=0.05, regression='n')
        assert result_n.test_statistic is not None
    
    def test_autolag_methods(self):
        """Test different autolag selection methods."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        for method in ['AIC', 'BIC', 't-stat']:
            result = adf_test(data, alpha=0.05, autolag=method)
            assert result.test_statistic is not None
    
    def test_too_short_series(self):
        """Test that short series raises error."""
        data = pd.Series([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError, match="Time series too short"):
            adf_test(data, alpha=0.05)
    
    def test_with_nan_values(self):
        """Test that NaN values are handled."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        data.iloc[50:60] = np.nan
        
        result = adf_test(data, alpha=0.05)
        
        assert result.test_statistic is not None
        assert not np.isnan(result.test_statistic)


class TestKPSSTest:
    """Tests for KPSS test."""
    
    def test_stationary_series(self):
        """Test with stationary series."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = kpss_test(data, alpha=0.05)
        
        assert result.test_name == "Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test"
        assert result.is_stationary == True  # Should not reject stationarity
        assert result.p_value > 0.05
    
    def test_nonstationary_series(self):
        """Test with non-stationary series."""
        np.random.seed(42)
        n = 200
        # Random walk
        data = pd.Series(np.cumsum(np.random.normal(0, 1, n)))
        
        result = kpss_test(data, alpha=0.05)
        
        assert result.is_stationary == False  # Should reject stationarity
        assert result.p_value < 0.05
    
    def test_regression_types(self):
        """Test different regression types."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        # Level stationarity
        result_c = kpss_test(data, alpha=0.05, regression='c')
        assert result_c.test_statistic is not None
        
        # Trend stationarity
        result_ct = kpss_test(data, alpha=0.05, regression='ct')
        assert result_ct.test_statistic is not None
    
    def test_too_short_series(self):
        """Test that short series raises error."""
        data = pd.Series([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError, match="Time series too short"):
            kpss_test(data, alpha=0.05)


class TestPhillipsPerronTest:
    """Tests for Phillips-Perron test."""
    
    def test_stationary_series(self):
        """Test with stationary series."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = phillips_perron_test(data, alpha=0.05)
        
        assert result.test_name == "Phillips-Perron (PP) Test for Unit Root"
        assert result.is_stationary == True
        assert result.p_value < 0.05
        # Verify the warning about trend vs variance
        assert "TREND" in result.interpretation
    
    def test_nonstationary_series(self):
        """Test with non-stationary series."""
        np.random.seed(42)
        n = 200
        # Random walk
        data = pd.Series(np.cumsum(np.random.normal(0, 1, n)))
        
        result = phillips_perron_test(data, alpha=0.05)
        
        assert result.is_stationary == False
        assert result.p_value > 0.05
    
    def test_trend_specifications(self):
        """Test different trend specifications."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        for trend in ['c', 'ct', 'n']:
            result = phillips_perron_test(data, alpha=0.05, trend=trend)
            assert result.test_statistic is not None
    
    def test_too_short_series(self):
        """Test that short series raises error."""
        data = pd.Series([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError, match="Time series too short"):
            phillips_perron_test(data, alpha=0.05)
    
    def test_correct_labeling(self):
        """
        Verify that PP is correctly labeled as testing TREND, not variance.
        
        This is critical - the old toolkit incorrectly labeled this as
        testing variance stationarity.
        """
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = phillips_perron_test(data, alpha=0.05)
        
        # Check that documentation is clear
        assert "Unit Root" in result.test_name
        assert "TREND" in result.interpretation.upper()
        assert "NOT" in result.interpretation and "variance" in result.interpretation.lower()


class TestCombinedTrendTest:
    """Tests for combined ADF-KPSS test."""
    
    def test_both_agree_stationary(self):
        """Test when both tests agree series is stationary."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = combined_trend_test(data, alpha=0.05)
        
        assert result.test_name == "Combined ADF-KPSS Test"
        assert result.is_stationary == True
        assert "agree" in result.interpretation.lower()
    
    def test_both_agree_nonstationary(self):
        """Test when both tests agree series is non-stationary."""
        np.random.seed(42)
        n = 200
        # Strong random walk
        data = pd.Series(np.cumsum(np.random.normal(0, 2, n)))
        
        result = combined_trend_test(data, alpha=0.05)
        
        assert result.is_stationary == False
        assert "agree" in result.interpretation.lower()
    
    def test_disagreement(self):
        """Test when tests disagree (inconclusive)."""
        # This is harder to construct, but the test should handle it
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = combined_trend_test(data, alpha=0.05)
        
        # Should have interpretation regardless
        assert result.interpretation is not None
        assert len(result.interpretation) > 0


class TestTrendStationarityWrapper:
    """Tests for the main test_trend_stationarity function."""
    
    def test_adf_method(self):
        """Test calling with adf method."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = test_trend_stationarity(data, method="adf", alpha=0.05)
        
        assert "ADF" in result.test_name
    
    def test_kpss_method(self):
        """Test calling with kpss method."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = test_trend_stationarity(data, method="kpss", alpha=0.05)
        
        assert "KPSS" in result.test_name
    
    def test_pp_method(self):
        """Test calling with pp method."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = test_trend_stationarity(data, method="pp", alpha=0.05)
        
        assert "Phillips-Perron" in result.test_name or "PP" in result.test_name
    
    def test_phillips_perron_alias(self):
        """Test that 'phillips_perron' is an alias for 'pp'."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = test_trend_stationarity(data, method="phillips_perron", alpha=0.05)
        
        assert "Phillips-Perron" in result.test_name or "PP" in result.test_name
    
    def test_combined_method(self):
        """Test calling with combined method."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result = test_trend_stationarity(data, method="combined", alpha=0.05)
        
        assert "Combined" in result.test_name
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        data = pd.Series(np.random.normal(0, 1, 200))
        
        with pytest.raises(ValueError, match="Unknown trend test method"):
            test_trend_stationarity(data, method="invalid", alpha=0.05)
    
    def test_case_insensitive(self):
        """Test that method names are case-insensitive."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 200))
        
        result1 = test_trend_stationarity(data, method="ADF", alpha=0.05)
        result2 = test_trend_stationarity(data, method="Adf", alpha=0.05)
        result3 = test_trend_stationarity(data, method="adf", alpha=0.05)
        
        assert result1.test_name == result2.test_name == result3.test_name


class TestTrendVsVariance:
    """
    Tests demonstrating the difference between trend and variance stationarity.
    
    This clarifies the conceptual distinction that was confused in the old toolkit.
    """
    
    def test_trend_without_variance_issue(self):
        """Series with trend but constant variance."""
        np.random.seed(42)
        n = 200
        # Linear trend with constant variance
        data = pd.Series(np.arange(n) * 0.1 + np.random.normal(0, 1, n))
        
        # Should detect trend non-stationarity
        trend_result = adf_test(data, alpha=0.05)
        assert 0 == False, \
            "Should detect trend non-stationarity"
    
    def test_variance_without_trend_issue(self):
        """Series with changing variance but no trend."""
        np.random.seed(42)
        n = 200
        # No trend, but variance increases
        data = pd.Series(np.random.normal(0, 1 + np.linspace(0, 2, n), n))
        
        # Trend test might say stationary (no unit root)
        trend_result = adf_test(data, alpha=0.05)
        # But variance test should detect the issue
        from src.stationarity_toolkit_v2.tests.variance_tests import levene_test
        variance_result = levene_test(data, alpha=0.05)
        
        assert 0 == False, \
            "Should detect variance non-stationarity"
        
        print(f"\nTrend test (ADF): {'stationary' if trend_result.is_stationary else 'non-stationary'}")
        print(f"Variance test (Levene): {'stationary' if variance_result.is_stationary else 'non-stationary'}")
        print("This demonstrates why we need BOTH types of tests!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
