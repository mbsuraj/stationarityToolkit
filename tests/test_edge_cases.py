import pytest
import pandas as pd
import numpy as np
from stationarity_toolkit.toolkit import StationarityToolkit


def test_invalid_input_type():
    toolkit = StationarityToolkit()
    with pytest.raises(ValueError, match="Input must be a pandas Series"):
        toolkit.detect([1, 2, 3, 4, 5])


def test_missing_datetime_index():
    toolkit = StationarityToolkit()
    series = pd.Series(np.random.randn(100))
    with pytest.raises(ValueError, match="Series must have a datetime index"):
        toolkit.detect(series)


def test_insufficient_observations():
    toolkit = StationarityToolkit()
    series = pd.Series(
        np.random.randn(30),
        index=pd.date_range('2020-01-01', periods=30, freq='D')
    )
    with pytest.raises(ValueError, match="Series must have at least 50 observations"):
        toolkit.detect(series)


def test_nan_handling(stationary_series):
    toolkit = StationarityToolkit()
    series_with_nan = stationary_series.copy()
    series_with_nan.iloc[10:15] = np.nan
    result = toolkit.detect(series_with_nan)
    assert result.trend_stationary is not None
    assert result.variance_stationary is not None
    assert result.seasonal_stationary is not None
