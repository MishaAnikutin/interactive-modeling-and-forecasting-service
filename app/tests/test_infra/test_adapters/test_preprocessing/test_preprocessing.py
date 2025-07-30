import numpy as np
import pandas as pd
import pytest

from src.core.application.preprocessing.preprocess_scheme import (
    DiffTransformation,
    LagTransformation,
    LogTransformation,
    PowTransformation,
    StandardTransformation,
    MinMaxTransformation,
    ExpSmoothTransformation,
    BoxCoxTransformation,
    FillMissingTransformation,
    MovingAverageTransformation,
)
from src.infrastructure.adapters.preprocessing.preprocess_factory import (
    PreprocessFactory,
)


@pytest.fixture
def sample_series() -> pd.Series:
    dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
    return pd.Series(range(1, 11), index=dates, name="y")


def test_diff(sample_series):
    res, _ = PreprocessFactory.apply(sample_series, DiffTransformation(diff_order=1))
    pd.testing.assert_series_equal(res, sample_series.diff(1))


def test_diff_context(sample_series):
    _, context = PreprocessFactory.apply(
        sample_series, DiffTransformation(diff_order=3)
    )
    assert context.first_values == list(sample_series[:3].values)


def test_diff_gt_size_of_data(sample_series):
    res, _ = PreprocessFactory.apply(
        sample_series, DiffTransformation(diff_order=sample_series.size + 10)
    )
    expected = pd.Series(np.nan, index=sample_series.index, name=sample_series.name)
    pd.testing.assert_series_equal(res, expected)


def test_lag(sample_series):
    res, _ = PreprocessFactory.apply(sample_series, LagTransformation(lag_order=2))
    pd.testing.assert_series_equal(res, sample_series.shift(2))


def test_log(sample_series):
    res, _ = PreprocessFactory.apply(sample_series, LogTransformation())
    pd.testing.assert_series_equal(
        res,
        np.log(
            sample_series,
        ),
    )

    ts = sample_series.copy()
    ts.iloc[[1, 5]] = -10
    res, _ = PreprocessFactory.apply(ts, LogTransformation())
    pd.testing.assert_series_equal(res, np.log(ts))


def test_pow(sample_series):
    res, _ = PreprocessFactory.apply(sample_series, PowTransformation(pow_order=2))
    pd.testing.assert_series_equal(res, sample_series**2)

    res, _ = PreprocessFactory.apply(sample_series, PowTransformation(pow_order=2.5))
    pd.testing.assert_series_equal(res, sample_series**2.5)


def test_standard(sample_series):
    res, _ = PreprocessFactory.apply(sample_series, StandardTransformation())
    assert np.isclose(res.mean(), 0.0)
    assert np.isclose(res.std(ddof=0), 1.0)


def test_standard_context(sample_series):
    _, context = PreprocessFactory.apply(sample_series, StandardTransformation())
    assert np.isclose(sample_series.mean(), context.init_mean)
    assert np.isclose(sample_series.std(ddof=0), context.init_std)


def test_minmax(sample_series):
    res, _ = PreprocessFactory.apply(sample_series, MinMaxTransformation())
    assert np.isclose(res.min(), 0.0)
    assert np.isclose(res.max(), 1.0)


def test_minmax_context(sample_series):
    _, context = PreprocessFactory.apply(sample_series, MinMaxTransformation())
    assert np.isclose(sample_series.min(), context.init_min)
    assert np.isclose(sample_series.max(), context.init_max)


def test_exp_smooth(sample_series):
    span = 3
    res, _ = PreprocessFactory.apply(sample_series, ExpSmoothTransformation(span=span))
    pd.testing.assert_series_equal(res, sample_series.ewm(span=span).mean())


def test_boxcox(sample_series):
    trans = BoxCoxTransformation(param=0.5)

    # базовый случай
    res, _ = PreprocessFactory.apply(sample_series, trans)
    assert res.notna().all()
    assert len(res) == len(sample_series)

    # случай с участком минусовых значений
    ts = sample_series.copy()
    ts.iloc[[1, 5]] = -10
    with pytest.raises(ValueError):
        res, _ = PreprocessFactory.apply(ts, trans)


@pytest.mark.parametrize("method", ["last", "backward", "mean", "median"])
def test_fillna(method, sample_series):
    ts = sample_series.copy()
    ts.iloc[[1, 5]] = np.nan
    res, _ = PreprocessFactory.apply(ts, FillMissingTransformation(method=method))
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
    res, _ = PreprocessFactory.apply(
        sample_series, MovingAverageTransformation(window=window)
    )
    pd.testing.assert_series_equal(
        res, sample_series.rolling(window=window, min_periods=1).mean()
    )
