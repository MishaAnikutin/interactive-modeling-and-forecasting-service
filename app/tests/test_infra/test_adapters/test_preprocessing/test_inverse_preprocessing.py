import numpy as np
import pandas as pd
import pytest

from src.core.application.preprocessing.preprocess_scheme import (
    DiffTransformation,
    LagTransformation,
    LogTransformation,
    StandardTransformation,
    MinMaxTransformation,
    InverseDiffTransformation,
    InverseLagTransformation,
    InverseStandardTransformation,
    InverseMinMaxTransformation,
    InverseLogTransformation,
)
from src.infrastructure.adapters.preprocessing.preprocess_factory import (
    PreprocessFactory,
)


@pytest.fixture
def sample_series() -> pd.Series:
    dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
    return pd.Series(np.arange(1.0, 11.0, 1.0), index=dates, name="y")


@pytest.mark.parametrize("diff_order", [1, 2, 3])
def test_diff_inverse(sample_series, diff_order):
    preprocessed_ts, context = PreprocessFactory.apply(
        sample_series, DiffTransformation(diff_order=diff_order)
    )

    restored_series = PreprocessFactory.inverse(
        preprocessed_ts,
        InverseDiffTransformation(
            diff_order=diff_order, first_values=context.first_values
        ),
    )

    pd.testing.assert_series_equal(restored_series, sample_series)


@pytest.mark.parametrize("lag_order", [1, 2, 3])
def test_lag(sample_series, lag_order):
    preprocessed_ts, context = PreprocessFactory.apply(
        sample_series, LagTransformation(lag_order=lag_order)
    )

    assert context is None

    restored_series = PreprocessFactory.inverse(
        preprocessed_ts,
        InverseLagTransformation(lag_order=lag_order),
    )

    pd.testing.assert_series_equal(
        restored_series[:-lag_order], sample_series[:-lag_order]
    )


def test_standard(sample_series):
    preprocessed_ts, context = PreprocessFactory.apply(
        sample_series, StandardTransformation()
    )

    restored_series = PreprocessFactory.inverse(
        preprocessed_ts,
        InverseStandardTransformation(
            init_mean=context.init_mean, init_std=context.init_std
        ),
    )

    restored_series = round(restored_series, 4)

    pd.testing.assert_series_equal(restored_series, sample_series)


def test_minmax(sample_series):
    preprocessed_ts, context = PreprocessFactory.apply(
        sample_series, MinMaxTransformation()
    )

    restored_series = PreprocessFactory.inverse(
        preprocessed_ts,
        InverseMinMaxTransformation(
            init_max=context.init_max, init_min=context.init_min
        ),
    )

    restored_series = round(restored_series, 4)

    pd.testing.assert_series_equal(restored_series, sample_series)


def test_inverse_sequence(sample_series):
    # minmax -> diff -> log
    preprocessed_series, minmax_context = PreprocessFactory.apply(
        sample_series, MinMaxTransformation()
    )
    preprocessed_series, diff_context = PreprocessFactory.apply(
        preprocessed_series, DiffTransformation(diff_order=3)
    )
    preprocessed_series, _ = PreprocessFactory.apply(
        preprocessed_series, LogTransformation()
    )

    # log -> diff -> minmax
    restored_series = PreprocessFactory.inverse(
        preprocessed_series, InverseLogTransformation()
    )
    restored_series = PreprocessFactory.inverse(
        restored_series,
        InverseDiffTransformation(diff_order=3, first_values=diff_context.first_values),
    )
    restored_series = PreprocessFactory.inverse(
        restored_series,
        InverseMinMaxTransformation(
            init_max=minmax_context.init_max, init_min=minmax_context.init_min
        ),
    )

    restored_series = round(restored_series)

    pd.testing.assert_series_equal(restored_series, sample_series)
