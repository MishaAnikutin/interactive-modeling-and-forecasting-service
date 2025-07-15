from datetime import datetime

import pandas as pd
import pytest

from src.core.application.building_model.schemas.lstm import LstmParams
from src.core.domain import FitParams, Timeseries
from src.infrastructure.adapters.modeling.neural_forecast import future_index
from tests.conftest import client, balance_ts, ca_ts, u_total_ts, balance, ipp_eu_ts


def process_fit_params(fit_params: FitParams) -> dict:
    return {
        "forecast_horizon": fit_params.forecast_horizon,
        "val_boundary": fit_params.val_boundary.strftime("%Y-%m-%d"),
        "train_boundary": fit_params.train_boundary.strftime("%Y-%m-%d"),
    }

def process_variable(ts: Timeseries) -> dict:
    return {
        "name": ts.name,
        "values": ts.values,
        "dates": [date.strftime("%Y-%m-%d") for date in ts.dates],
        "data_frequency": ts.data_frequency,
    }

def from_pd_stamp_to_datetime(ts: list[pd.Timestamp]) -> list[str]:
    return [date.strftime("%Y-%m-%d") for date in ts]

def delete_timestamp(ts: list[str]) -> list[str]:
    return [date.replace("T00:00:00", "") if "T00:00:00" in date else date for date in ts]

def validate_no_exog_result(
        received_data: dict,
        dependent_variables,
        data,
        fit_params,
):
    # проверяем прогнозы
    forecasts = received_data['forecasts']

    train_predict = forecasts['train_predict']
    test_predict = forecasts['test_predict']
    forecast = forecasts['forecast']

    assert delete_timestamp(
        train_predict['dates'] + test_predict['dates']
    ) == data['dependent_variables']['dates'], \
        "Не сходятся даты в предикте и в исходных данных"
    if fit_params.forecast_horizon > 0:
        assert from_pd_stamp_to_datetime(future_index(
            last_dt=pd.to_datetime(dependent_variables.dates[-1]),
            data_frequency=dependent_variables.data_frequency,
            periods=fit_params.forecast_horizon,
        ).tolist()) == delete_timestamp(forecast['dates'])
    else:
        assert forecast is None

    # проверяем метрики
    assert received_data['model_metrics']['train_metrics'], "Train-метрики не рассчитаны"
    assert received_data['model_metrics']['test_metrics'], "Test-метрики не рассчитаны"

    assert received_data['weight_path'], "Путь к весам пуст"

    metrics = received_data['model_metrics']['test_metrics']
    types = tuple(m['type'] for m in metrics)
    assert types == ("RMSE", "MAPE", "R2")

def validate_empty_test_data(
        received_data: dict,
        dependent_variables,
        data,
        fit_params,
):
    # проверяем прогнозы
    forecasts = received_data['forecasts']

    train_predict = forecasts['train_predict']
    forecast = forecasts['forecast']

    assert delete_timestamp(train_predict['dates']) == data['dependent_variables']['dates']
    assert from_pd_stamp_to_datetime(future_index(
        last_dt=pd.to_datetime(dependent_variables.dates[-1]),
        data_frequency=dependent_variables.data_frequency,
        periods=fit_params.forecast_horizon,
    ).tolist()) == delete_timestamp(forecast['dates'])

    # проверяем метрики
    assert received_data['model_metrics']['train_metrics'], "Train-метрики не рассчитаны"
    assert received_data['model_metrics']['test_metrics'] is None, "Test-метрики рассчитаны"

    assert received_data['weight_path'], "Путь к весам пуст"

    metrics = received_data['model_metrics']['train_metrics']
    types = tuple(m['type'] for m in metrics)
    assert types == ("RMSE", "MAPE", "R2")


@pytest.mark.parametrize(
    "lstm_params, fit_params, dependent_variables",
    [
        ( # самый базовый кейс с дефолтными значениями параметров
            LstmParams(),
            FitParams(),
            Timeseries()
        ),
        ( # месячные данные
            LstmParams(),
            FitParams(
                train_boundary=datetime(2020, 12, 31),
                val_boundary=datetime(2024, 1,31)
            ),
            balance_ts()
        ),
        ( # квартальные данные
            LstmParams(),
            FitParams(
                train_boundary=datetime(2019, 6, 30),
                val_boundary=datetime(2021, 6, 30),
                forecast_horizon=3,
            ),
            ca_ts()
        ),
        ( # годовые данные
            LstmParams(),
            FitParams(
                train_boundary=datetime(2012, 12, 31),
                val_boundary=datetime(2021, 12, 31),
                forecast_horizon=3,
            ),
            u_total_ts()
        )
    ]
)
def test_lstm_fit_without_exog(
    lstm_params,
    fit_params,
    dependent_variables,
    client
):
    data = dict(
        dependent_variables=process_variable(dependent_variables),
        explanatory_variables=None,
        hyperparameters=lstm_params.model_dump(),
        fit_params=process_fit_params(fit_params),
    )
    result = client.post(
        url='/api/v1/building_model/lstm/fit',
        json=data
    )

    received_data = result.json()
    assert result.status_code == 200, received_data

    validate_no_exog_result(received_data, dependent_variables, data, fit_params)