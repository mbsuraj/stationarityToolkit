import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def stationary_series() -> pd.Series:
    """White noise series with DatetimeIndex."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    return pd.Series(np.random.randn(n), index=dates)


@pytest.fixture
def trend_series() -> pd.Series:
    """Linear trend series with DatetimeIndex."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    return pd.Series(np.arange(n) + np.random.randn(n) * 0.5, index=dates)


@pytest.fixture
def variance_series() -> pd.Series:
    """Heteroskedastic variance series with DatetimeIndex."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    noise = np.random.randn(n)
    return pd.Series(noise * np.sqrt(np.arange(1, n + 1)), index=dates)


@pytest.fixture
def seasonal_series() -> pd.Series:
    """Periodic pattern series with DatetimeIndex."""
    np.random.seed(42)
    n = 100
    period = 12
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    t = np.arange(n)
    return pd.Series(np.sin(2 * np.pi * t / period) + np.random.randn(n) * 0.3, index=dates)


@pytest.fixture
def combined_series() -> pd.Series:
    """Combined non-stationarity: trend + seasonal + heteroskedastic variance."""
    np.random.seed(42)
    n = 100
    period = 12
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    t = np.arange(n)
    trend = 0.5 * t
    seasonal = 2 * np.sin(2 * np.pi * t / period)
    noise = np.random.randn(n) * np.sqrt(t + 1)
    return pd.Series(trend + seasonal + noise, index=dates)
