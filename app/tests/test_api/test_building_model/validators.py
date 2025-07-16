import pandas as pd

from src.core.domain import FitParams, Timeseries
from src.infrastructure.adapters.modeling.neural_forecast import future_index


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
    validation_predict = forecasts['validation_predict']
    test_predict = forecasts['test_predict']
    forecast = forecasts['forecast']

    assert delete_timestamp(
        train_predict['dates'] + validation_predict['dates'] + test_predict['dates']
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
    if validation_predict['dates']:
        assert received_data['model_metrics']['val_metrics'], "Val-метрики не рассчитаны"
    else:
        assert not received_data['model_metrics']['val_metrics'], "Val-метрики рассчитаны, хотя не должны"
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
    validation_predict = forecasts['validation_predict']
    forecast = forecasts['forecast']

    assert train_predict['dates'], "Почему-то прогноз на обучающей пустой"
    assert validation_predict['dates'], "Почему-то валидационный прогноз пустой"

    assert delete_timestamp(
        train_predict['dates'] + validation_predict['dates']
    ) == data['dependent_variables']['dates']
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
    assert received_data['model_metrics']['test_metrics'] is None, "Test-метрики рассчитаны"
    assert received_data['model_metrics']['val_metrics'], "Val-метрики не рассчитаны"

    assert received_data['weight_path'], "Путь к весам пуст"

    metrics = received_data['model_metrics']['train_metrics']
    types = tuple(m['type'] for m in metrics)
    assert types == ("RMSE", "MAPE", "R2")

def validate_empty_val_data(
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

    assert train_predict['dates'], "Почему-то прогноз на обучающей пустой"
    assert test_predict['dates'], "Почему-то прогноз на тестовой выборке пустой"

    assert delete_timestamp(train_predict['dates'] + test_predict['dates']) == data['dependent_variables']['dates']
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
    assert received_data['model_metrics']['val_metrics'] is None, "Val-метрики рассчитаны"
    assert received_data['model_metrics']['test_metrics'], "Test-метрики не рассчитаны"

    assert received_data['weight_path'], "Путь к весам пуст"

    metrics = received_data['model_metrics']['train_metrics']
    types = tuple(m['type'] for m in metrics)
    assert types == ("RMSE", "MAPE", "R2")


def validate_only_train_data(
        received_data: dict,
        dependent_variables,
        data,
        fit_params,
):
    # проверяем прогнозы
    forecasts = received_data['forecasts']

    train_predict = forecasts['train_predict']
    forecast = forecasts['forecast']

    assert train_predict['dates'], "Почему-то прогноз на обучающей пустой"

    assert delete_timestamp(train_predict['dates']) == data['dependent_variables']['dates']
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
    assert received_data['model_metrics']['test_metrics'] is None, "Test-метрики рассчитаны"
    assert received_data['model_metrics']['val_metrics'] is None, "Val-метрики рассчитаны"

    assert received_data['weight_path'], "Путь к весам пуст"

    metrics = received_data['model_metrics']['train_metrics']
    types = tuple(m['type'] for m in metrics)
    assert types == ("RMSE", "MAPE", "R2")