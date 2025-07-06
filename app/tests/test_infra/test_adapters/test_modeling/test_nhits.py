from datetime import datetime

import pandas as pd
import pytest

from src.core.application.building_model.schemas.nhits import NhitsParams
from src.core.domain import FitParams, DataFrequency, Timeseries
from tests.conftest import nhits_adapter, ipp_eu, ipp_eu_ts, u_men, u_women, u_total, ts_alignment, balance

@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "nhits_params, fit_params",
    [
        (
            NhitsParams(
                max_steps=30,
                early_stop_patience_steps=3,
                val_check_steps=50,
                learning_rate=1e-3,
                scaler_type="robust",
            ),
            FitParams(
                train_boundary=datetime(2014, 6, 30),
                val_boundary=datetime(2018, 5, 31),
                forecast_horizon=20,
            ),
        ),
        (
            NhitsParams(
                max_steps=500,
                early_stop_patience_steps=50,
                val_check_steps=200,
                learning_rate=5e-4,
                scaler_type="robust",
            ),
            FitParams(
                train_boundary=datetime(2013, 12, 31),
                val_boundary=datetime(2017, 12, 31),
                forecast_horizon=24,
            ),
        ),
    ]
)
def test_nhits_fit_without_exog_month_frequency(
    nhits_params,
    fit_params,
    nhits_adapter,
    ipp_eu
) -> None:
    result = nhits_adapter.fit(
        target=ipp_eu,
        exog=None,
        nhits_params=nhits_params,
        fit_params=fit_params,
        data_frequency=DataFrequency.month
    )

    assert result.forecasts.train_predict.dates, "Пустой train-прогноз"
    assert result.forecasts.test_predict.dates, "Пустой test-прогноз"

    assert result.model_metrics.train_metrics, "Train-метрики не рассчитаны"
    assert result.model_metrics.test_metrics, "Test-метрики не рассчитаны"

    assert result.weight_path, "Путь к весам пуст"

    metrics = result.model_metrics.test_metrics
    for m in metrics:
        if m.type == "R2":
            assert m.value > -1, "Что-то не так с моделью, R2 меньше -1, походу был сильный шок в данных"
        elif m.type == "MAPE":
            assert m.value < 1
        elif m.type == "RMSE":
            assert m.value > 0.5, "Что-то не так с моделью, RMSE маленькое"

    # Проверка прогнозов
    train_predict_len = len(result.forecasts.train_predict.dates)
    test_predict_len = len(result.forecasts.test_predict.dates)
    future_predict_len = len(result.forecasts.forecast.dates)
    assert train_predict_len + test_predict_len == ipp_eu.shape[0]
    assert future_predict_len == fit_params.forecast_horizon


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "nhits_params, fit_params",
    [
        (
            NhitsParams(
                max_steps=30,
                early_stop_patience_steps=3,
                val_check_steps=50,
                learning_rate=1e-3,
                scaler_type="robust",
            ),
            FitParams(
                train_boundary=datetime(2012, 12, 31),
                val_boundary=datetime(2021, 12, 31),
                forecast_horizon=3,
            ),
        ),
    ]
)
def test_nhits_fit_with_two_exog_year_frequency(
    nhits_params,
    fit_params,
    nhits_adapter,
    ts_alignment,
    u_total,
    u_men,
    u_women,
):
    target = pd.Series(
        index=u_total.dates,
        data=u_total.values,
        name=u_total.name
    )
    exog = ts_alignment.compare(
        timeseries_list=[u_men, u_women],
        target=u_total,
    )
    result = nhits_adapter.fit(
        target=target,
        exog=exog,
        nhits_params=nhits_params,
        fit_params=fit_params,
        data_frequency=u_total.data_frequency
    )

    assert result.forecasts.train_predict.dates, "Пустой train-прогноз"
    assert result.forecasts.test_predict.dates, "Пустой test-прогноз"

    assert result.model_metrics.train_metrics, "Train-метрики не рассчитаны"
    assert result.model_metrics.test_metrics, "Test-метрики не рассчитаны"

    assert result.weight_path, "Путь к весам пуст"

    metrics = result.model_metrics.test_metrics
    types = tuple(m.type for m in metrics)
    assert types == nhits_adapter.metrics

    # Проверка прогнозов
    train_predict_len = len(result.forecasts.train_predict.dates)
    test_predict_len = len(result.forecasts.test_predict.dates)
    future_predict_len = len(result.forecasts.forecast.dates)
    assert train_predict_len + test_predict_len == target.shape[0]
    assert future_predict_len == fit_params.forecast_horizon


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "nhits_params, fit_params",
    [
        (
            NhitsParams(
                max_steps=30,
                early_stop_patience_steps=3,
                val_check_steps=50,
                learning_rate=1e-3,
                scaler_type="robust",
            ),
            FitParams(
                train_boundary=datetime(2018, 5,31),
                val_boundary=datetime(2019, 7, 31),
                forecast_horizon=3,
            ),
        ),
    ]
)
def test_nhits_fit_with_one_exog_month_frequency(
    nhits_params,
    fit_params,
    nhits_adapter,
    ipp_eu,
    ipp_eu_ts,
    balance,
    ts_alignment
):
    aligned_df = ts_alignment.compare(
        timeseries_list=[balance],
        target=ipp_eu_ts,
    )

    target = aligned_df[ipp_eu_ts.name]

    assert type(target) == pd.Series
    assert aligned_df.columns.to_list() == [ipp_eu_ts.name, balance.name]

    exog = aligned_df.drop(columns=[ipp_eu_ts.name])

    result = nhits_adapter.fit(
        target=target,
        exog=exog,
        nhits_params=nhits_params,
        fit_params=fit_params,
        data_frequency=ipp_eu_ts.data_frequency,
    )

    assert result.forecasts.train_predict.dates, "Пустой train-прогноз"
    assert result.forecasts.test_predict.dates, "Пустой test-прогноз"

    assert result.model_metrics.train_metrics, "Train-метрики не рассчитаны"
    assert result.model_metrics.test_metrics, "Test-метрики не рассчитаны"

    assert result.weight_path, "Путь к весам пуст"

    metrics = result.model_metrics.test_metrics
    types = tuple(m.type for m in metrics)
    assert types == nhits_adapter.metrics

    # Проверка прогнозов
    train_predict_len = len(result.forecasts.train_predict.dates)
    test_predict_len = len(result.forecasts.test_predict.dates)
    future_predict_len = len(result.forecasts.forecast.dates)
    assert train_predict_len + test_predict_len == target.shape[0]
    assert future_predict_len == fit_params.forecast_horizon