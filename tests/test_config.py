"""
Unit tests for configuration validation.
"""

import pytest
from src.stationarity_toolkit_v2.config import StationarityConfig


class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StationarityConfig()
        
        assert config.alpha == 0.05
        assert config.seasonal_period is None
        assert config.max_differencing == 2
        assert config.variance_test == "levene"
        assert config.trend_test == "adf"
        assert config.auto_detect_seasonality is True
        assert config.min_observations == 50
        assert config.verbose is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = StationarityConfig(
            alpha=0.01,
            seasonal_period=12,
            max_differencing=3,
            variance_test="bartlett",
            trend_test="kpss",
            auto_detect_seasonality=False,
            min_observations=100,
            verbose=True
        )
        
        assert config.alpha == 0.01
        assert config.seasonal_period == 12
        assert config.max_differencing == 3
        assert config.variance_test == "bartlett"
        assert config.trend_test == "kpss"
        assert config.auto_detect_seasonality is False
        assert config.min_observations == 100
        assert config.verbose is True
    
    def test_invalid_alpha_too_low(self):
        """Test that alpha <= 0 raises error."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            StationarityConfig(alpha=0)
        
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            StationarityConfig(alpha=-0.1)
    
    def test_invalid_alpha_too_high(self):
        """Test that alpha >= 1 raises error."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            StationarityConfig(alpha=1)
        
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            StationarityConfig(alpha=1.5)
    
    def test_invalid_seasonal_period(self):
        """Test that seasonal_period < 2 raises error."""
        with pytest.raises(ValueError, match="seasonal_period must be >= 2"):
            StationarityConfig(seasonal_period=1)
        
        with pytest.raises(ValueError, match="seasonal_period must be >= 2"):
            StationarityConfig(seasonal_period=0)
    
    def test_valid_seasonal_period(self):
        """Test that valid seasonal periods work."""
        for period in [2, 4, 7, 12, 52, 365]:
            config = StationarityConfig(seasonal_period=period)
            assert config.seasonal_period == period
    
    def test_invalid_max_differencing(self):
        """Test that negative max_differencing raises error."""
        with pytest.raises(ValueError, match="max_differencing must be >= 0"):
            StationarityConfig(max_differencing=-1)
    
    def test_valid_max_differencing(self):
        """Test that valid max_differencing values work."""
        for order in [0, 1, 2, 3]:
            config = StationarityConfig(max_differencing=order)
            assert config.max_differencing == order
    
    def test_invalid_variance_test(self):
        """Test that invalid variance test raises error."""
        with pytest.raises(ValueError, match="variance_test must be one of"):
            StationarityConfig(variance_test="invalid")
    
    def test_valid_variance_tests(self):
        """Test that all valid variance tests work."""
        for test in ["levene", "bartlett", "white", "arch"]:
            config = StationarityConfig(variance_test=test)
            assert config.variance_test == test
    
    def test_invalid_trend_test(self):
        """Test that invalid trend test raises error."""
        with pytest.raises(ValueError, match="trend_test must be one of"):
            StationarityConfig(trend_test="invalid")
    
    def test_valid_trend_tests(self):
        """Test that all valid trend tests work."""
        for test in ["adf", "kpss", "pp"]:
            config = StationarityConfig(trend_test=test)
            assert config.trend_test == test
    
    def test_invalid_transformation_method(self):
        """Test that invalid transformation method raises error."""
        with pytest.raises(ValueError, match="Invalid transformation methods"):
            StationarityConfig(transformation_methods=["log", "invalid"])
    
    def test_valid_transformation_methods(self):
        """Test that valid transformation methods work."""
        methods = ["log", "sqrt", "boxcox"]
        config = StationarityConfig(transformation_methods=methods)
        assert config.transformation_methods == methods
    
    def test_transformation_methods_with_none(self):
        """Test that 'none' is a valid transformation method."""
        config = StationarityConfig(transformation_methods=["log", "none"])
        assert "none" in config.transformation_methods


class TestConfigUsage:
    """Tests for using configuration in practice."""
    
    def test_config_immutability(self):
        """Test that config values can be accessed."""
        config = StationarityConfig(alpha=0.05)
        
        # Should be able to read
        assert config.alpha == 0.05
        
        # Should be able to modify (dataclass is mutable by default)
        config.alpha = 0.01
        assert config.alpha == 0.01
    
    def test_config_with_toolkit(self):
        """Test that config works with toolkit."""
        from src.stationarity_toolkit_v2 import StationarityToolkit
        
        config = StationarityConfig(
            alpha=0.01,
            variance_test="bartlett",
            trend_test="kpss"
        )
        
        toolkit = StationarityToolkit(config=config)
        
        assert toolkit.config.alpha == 0.01
        assert toolkit.config.variance_test == "bartlett"
        assert toolkit.config.trend_test == "kpss"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
