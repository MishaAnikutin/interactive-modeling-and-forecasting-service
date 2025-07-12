import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from src.core.application.preprocessing.preprocess_scheme import (
    DiffTransformation,
    LagTransformation,
    LogTransformation,
    PowTransformation,
    NormalizeTransformation,
    ExpSmoothTransformation,
    BoxCoxTransformation,
    FillMissingTransformation,
    MovingAverageTransformation,
)
from src.infrastructure.adapters.preprocessing.preprocess_factory import PreprocessFactory


@pytest.fixture
def sample_series() -> pd.Series:
    dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
    return pd.Series(range(1, 11), index=dates, name="y")


def test_diff(sample_series):
    res = PreprocessFactory.apply(sample_series, DiffTransformation(diff_order=1))
    pd.testing.assert_series_equal(res, sample_series.diff(1))


def test_diff_gt_size_of_data(sample_series):
    res = PreprocessFactory.apply(sample_series, DiffTransformation(diff_order=sample_series.size + 10))
    expected = pd.Series(np.nan, index=sample_series.index, name=sample_series.name)
    pd.testing.assert_series_equal(res, expected)


def test_lag(sample_series):
    res = PreprocessFactory.apply(sample_series, LagTransformation(lag_order=2))
    pd.testing.assert_series_equal(res, sample_series.shift(2))


def test_log(sample_series):
    res = PreprocessFactory.apply(sample_series, LogTransformation())
    pd.testing.assert_series_equal(res, np.log(sample_series,))

    ts = sample_series.copy()
    ts.iloc[[1, 5]] = -10
    res = PreprocessFactory.apply(ts, LogTransformation())
    pd.testing.assert_series_equal(res, np.log(ts))


def test_pow(sample_series):
    res = PreprocessFactory.apply(sample_series, PowTransformation(pow_order=2))
    pd.testing.assert_series_equal(res, sample_series ** 2)

    res = PreprocessFactory.apply(sample_series, PowTransformation(pow_order=2.5))
    pd.testing.assert_series_equal(res, sample_series ** 2.5)


@pytest.mark.parametrize("method", ["standard", "minmax"])
def test_normalize(sample_series, method):
    res = PreprocessFactory.apply(sample_series, NormalizeTransformation(method=method))
    if method == "standard":
        assert np.isclose(res.mean(), 0.0)
        assert np.isclose(res.std(ddof=0), 1.0)
    else:
        assert np.isclose(res.min(), 0.0)
        assert np.isclose(res.max(), 1.0)


def test_exp_smooth(sample_series):
    span = 3
    res = PreprocessFactory.apply(sample_series, ExpSmoothTransformation(span=span))
    pd.testing.assert_series_equal(res, sample_series.ewm(span=span).mean())


def test_boxcox(sample_series):
    trans = BoxCoxTransformation(param=0.5)

    # базовый случай
    res = PreprocessFactory.apply(sample_series, trans)
    assert res.notna().all()
    assert len(res) == len(sample_series)

    # случай с участком минусовых значений
    ts = sample_series.copy()
    ts.iloc[[1, 5]] = -10
    with pytest.raises(HTTPException) as exc:
        PreprocessFactory.apply(ts, trans)
    assert exc.value.status_code == 400
    assert "Все элементы ряда должны быть положительные" in exc.value.detail


@pytest.mark.parametrize("method", ["last", "backward", "mean", "median"])
def test_fillna(method, sample_series):
    ts = sample_series.copy()
    ts.iloc[[1, 5]] = np.nan
    res = PreprocessFactory.apply(ts, FillMissingTransformation(method=method))
    assert not res.isna().any()
    if method == "last":
        assert np.isclose(res.iloc[1], ts.ffill().iloc[1])
    elif method == "backward":
        assert np.isclose(res.iloc[1], ts.bfill().iloc[1])
    elif method == "mean":
        assert np.isclose(res.iloc[1], ts.mean())
    else:
        assert np.isclose(res.iloc[1], ts.median())


def test_moving_average(sample_series):
    window = 4
    res = PreprocessFactory.apply(sample_series, MovingAverageTransformation(window=window))
    pd.testing.assert_series_equal(res, sample_series.rolling(window=window, min_periods=1).mean())