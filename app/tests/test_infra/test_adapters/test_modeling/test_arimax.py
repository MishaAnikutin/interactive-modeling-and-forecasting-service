from datetime import datetime

import pandas as pd
import pytest

from src.core.application.building_model.schemas.arimax import ArimaxParams
from src.core.domain import FitParams, DataFrequency
from src.infrastructure.adapters.modeling import ArimaxAdapter
from tests.conftest import arimax_adapter, ts_alignment


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "arimax_params, fit_params",
    [
        (
            ArimaxParams(
                p=1,
                d=1,
                q=0
            ),
            FitParams(
                train_boundary=datetime(2020, 12, 31),
                val_boundary=datetime(2022, 12, 31),
                forecast_horizon=3,
            ),
        ),
    ]
)
def test_arimax_fit_without_exog_month_frequency(
    arimax_params: ArimaxParams,
    fit_params: FitParams,
    arimax_adapter: ArimaxAdapter,
    ipc
) -> None:
    target = pd.Series(
        index=ipc.dates,
        data=ipc.values,
        name=ipc.name
    )

    # Train:      с 1999-01-31 по 2020-12-31
    # Validation: с 2021-01-31 по 2022-12-31
    # Test:       с 2023-01-31 по 2025-05-31

    result = arimax_adapter.fit(
        target=target,
        exog=None,
        arimax_params=arimax_params,
        fit_params=fit_params,
        data_frequency=DataFrequency.month
    )

    assert result.forecasts.train_predict.dates, "Пустой train-прогноз"
    assert result.forecasts.test_predict.dates, "Пустой test-прогноз"

    assert result.model_metrics.train_metrics, "Train-метрики не рассчитаны"
    assert result.model_metrics.test_metrics, "Test-метрики не рассчитаны"

    metrics = result.model_metrics.test_metrics

    for m in metrics:
        assert m.value is not None
        assert isinstance(m.value, float)

    train_predict_len = len(result.forecasts.train_predict.dates)
    test_predict_len = len(result.forecasts.test_predict.dates)
    val_predict_len = len(result.forecasts.validation_predict.dates)
    future_predict_len = len(result.forecasts.forecast.dates)

    assert train_predict_len + val_predict_len + test_predict_len == target.shape[0], f"длина трейна + теста не равна длины таргета: {train_predict_len + test_predict_len = }, {target.shape[0] = }"
    assert future_predict_len == fit_params.forecast_horizon, f'длина вневыборочного прогноза не равна окну прогнозирования: {future_predict_len = }, {fit_params.forecast_horizon = }'


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "arimax_params, fit_params",
    [
        (
            ArimaxParams(
                p=1,
                d=1,
                q=0
            ),
            FitParams(
                train_boundary=datetime(2012, 12, 31),
                val_boundary=datetime(2021, 12, 31),
                forecast_horizon=3,
            ),
        ),
    ]
)
def test_arimax_fit_with_exog_month_frequency(
    arimax_params,
    fit_params,
    arimax_adapter,
    ts_alignment,
    ipc,
    brent_oil
):
    df = ts_alignment.compare(
        timeseries_list=[brent_oil],
        target=ipc,
    )

    target = df[ipc.name]
    exog = df.drop(columns=[ipc.name])

    result = arimax_adapter.fit(
        target=target,
        exog=exog,
        arimax_params=arimax_params,
        fit_params=fit_params,
        data_frequency=DataFrequency.month
    )

    assert result.forecasts.train_predict.dates, "Пустой train-прогноз"
    assert result.forecasts.test_predict.dates, "Пустой test-прогноз"

    assert result.model_metrics.train_metrics, "Train-метрики не рассчитаны"
    assert result.model_metrics.test_metrics, "Test-метрики не рассчитаны"

    metrics = result.model_metrics.test_metrics

    # Проверка прогнозов
    train_predict_len = len(result.forecasts.train_predict.dates)
    test_predict_len = len(result.forecasts.test_predict.dates)
    val_predict_len = len(result.forecasts.validation_predict.dates)

    assert train_predict_len + val_predict_len + test_predict_len == target.shape[
        0], f"длина трейна + теста не равна длины таргета: {train_predict_len + test_predict_len = }, {target.shape[0] = }"
    assert result.forecasts.forecast is None, f'вневыборочный прогноз не пустой'

