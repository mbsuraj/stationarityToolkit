import pytest
from stationarity_toolkit.tests.trend import (
    adf_test,
    kpss_test,
    phillips_perron_test,
    zivot_andrews_test
)


def test_adf_stationary(stationary_series):
    result = adf_test(stationary_series)
    assert result.is_stationary is True


def test_adf_trend(trend_series):
    result = adf_test(trend_series)
    assert result.is_stationary is False


def test_kpss_stationary(stationary_series):
    result = kpss_test(stationary_series)
    assert result.is_stationary is True


def test_kpss_trend(trend_series):
    result = kpss_test(trend_series)
    assert result.is_stationary is False


def test_phillips_perron_stationary(stationary_series):
    result = phillips_perron_test(stationary_series)
    assert result.is_stationary is True


def test_phillips_perron_trend(trend_series):
    result = phillips_perron_test(trend_series)
    assert result.is_stationary is False


def test_zivot_andrews_stationary(stationary_series):
    result = zivot_andrews_test(stationary_series)
    assert result.is_stationary is True


def test_zivot_andrews_trend(trend_series):
    # ZA detects discrete structural breaks, not smooth trends.
    # It finds spurious breaks in linear trends and marks as stationary.
    result = zivot_andrews_test(trend_series)
    assert result.is_stationary is True


def test_testresult_structure(stationary_series):
    result = adf_test(stationary_series)
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
