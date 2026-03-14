import pytest
from stationarity_toolkit.toolkit import StationarityToolkit


def test_detect_stationary_series(stationary_series):
    toolkit = StationarityToolkit()
    result = toolkit.detect(stationary_series)
    assert result.trend_stationary is True
    assert result.variance_stationary is True
    assert result.seasonal_stationary is True


def test_detect_trend_series(trend_series):
    toolkit = StationarityToolkit()
    result = toolkit.detect(trend_series)
    assert result.trend_stationary is False


def test_detect_variance_series(variance_series):
    toolkit = StationarityToolkit()
    result = toolkit.detect(variance_series)
    assert result.variance_stationary is False


def test_detect_seasonal_series(seasonal_series):
    toolkit = StationarityToolkit()
    result = toolkit.detect(seasonal_series)
    assert result.seasonal_stationary is False


def test_verbosity_minimal(stationary_series):
    toolkit = StationarityToolkit()
    result = toolkit.detect(stationary_series, verbosity='minimal')
    assert result.tests == {}


def test_verbosity_detailed(stationary_series):
    toolkit = StationarityToolkit()
    result = toolkit.detect(stationary_series, verbosity='detailed')
    assert 'trend' in result.tests
    assert 'variance' in result.tests
    assert 'seasonal' in result.tests


def test_alpha_parameter(stationary_series):
    toolkit = StationarityToolkit(alpha=0.01)
    result = toolkit.detect(stationary_series)
    assert toolkit.alpha == 0.01
