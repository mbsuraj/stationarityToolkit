import pytest
from stationarity_toolkit.tests.seasonal import acf_peak_test, stl_test


def test_acf_peak_stationary(stationary_series):
    result = acf_peak_test(stationary_series)
    assert result.is_stationary == True


def test_acf_peak_seasonal(seasonal_series):
    result = acf_peak_test(seasonal_series)
    assert result.is_stationary == False


def test_stl_stationary(stationary_series):
    result = stl_test(stationary_series)
    assert result.is_stationary == True


def test_stl_seasonal(seasonal_series):
    result = stl_test(seasonal_series, period=12)
    assert result.is_stationary == False


def test_testresult_structure(stationary_series):
    result = acf_peak_test(stationary_series)
    assert hasattr(result, 'test_name')
    assert hasattr(result, 'statistic')
    assert hasattr(result, 'p_value')
    assert hasattr(result, 'is_stationary')
    assert hasattr(result, 'interpretation')
    assert hasattr(result, 'educational_note')
    assert isinstance(result.test_name, str)
    assert isinstance(result.statistic, float)
    assert isinstance(result.p_value, float)
    assert isinstance(result.is_stationary, bool)
    assert isinstance(result.interpretation, str)
    assert isinstance(result.educational_note, str)
