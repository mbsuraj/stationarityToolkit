import pytest
from stationarity_toolkit.tests.variance import (
    levene_test,
    bartlett_test,
    white_test,
    arch_test
)


def test_levene_stationary(stationary_series):
    result = levene_test(stationary_series)
    assert result.is_stationary == True
    assert result.test_name == "Levene's Test for Variance Homogeneity"
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.interpretation is not None
    assert result.educational_note is not None


def test_levene_variance(variance_series):
    result = levene_test(variance_series)
    assert result.is_stationary == False


def test_bartlett_stationary(stationary_series):
    result = bartlett_test(stationary_series)
    assert result.is_stationary == True
    assert result.test_name == "Bartlett's Test for Variance Homogeneity"
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.interpretation is not None
    assert result.educational_note is not None


def test_bartlett_variance(variance_series):
    result = bartlett_test(variance_series)
    assert result.is_stationary == False


def test_white_stationary(stationary_series):
    result = white_test(stationary_series)
    assert result.is_stationary == True
    assert result.test_name == "White's Test for Heteroskedasticity"
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.interpretation is not None
    assert result.educational_note is not None


def test_white_variance(variance_series):
    result = white_test(variance_series)
    assert result.is_stationary == False


def test_arch_stationary(stationary_series):
    result = arch_test(stationary_series)
    assert result.is_stationary == True
    assert result.test_name == "ARCH Test for Conditional Heteroskedasticity"
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.interpretation is not None
    assert result.educational_note is not None


def test_arch_variance(variance_series):
    result = arch_test(variance_series)
    # ARCH detects volatility clustering, not smooth variance changes
    # Smooth variance increase is correctly identified as stationary by ARCH
