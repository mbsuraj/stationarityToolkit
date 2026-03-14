import pandas as pd
import numpy as np
from stationarity_toolkit.utils import get_contextual_periods


def _make_series(n, freq):
    return pd.Series(np.random.randn(n), index=pd.date_range('2020-01-01', periods=n, freq=freq))


def test_daily_periods():
    assert get_contextual_periods(_make_series(100, 'D')) == [7, 30, 365]


def test_weekly_periods():
    assert get_contextual_periods(_make_series(100, 'W')) == [52]


def test_weekly_monday_periods():
    assert get_contextual_periods(_make_series(100, 'W-MON')) == [52]


def test_hourly_periods():
    assert get_contextual_periods(_make_series(100, 'h')) == [24, 168]


def test_monthly_periods():
    assert get_contextual_periods(_make_series(100, 'MS')) == [3, 6, 12]


def test_quarterly_periods():
    assert get_contextual_periods(_make_series(100, 'QS')) == [4]


def test_no_freq_fallback():
    s = pd.Series(np.random.randn(10), index=pd.to_datetime([
        '2020-01-01', '2020-01-03', '2020-01-07', '2020-01-08',
        '2020-01-15', '2020-01-20', '2020-02-01', '2020-02-10',
        '2020-03-01', '2020-04-01'
    ]))
    assert get_contextual_periods(s) == [7, 12, 30, 52, 365]


def test_business_day_periods():
    assert get_contextual_periods(_make_series(100, 'B')) == [7, 30, 365]
