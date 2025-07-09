import json
import random
from datetime import datetime
import pandas as pd
import pytest

from src.core.application.building_model.schemas.nhits import NhitsParams, PoolingMode, InterpMode, LossEnum, \
    ActivationType, ScalerType
from src.core.domain import FitParams, Timeseries
from src.infrastructure.adapters.modeling.neural_forecast import future_index
from tests.conftest import client, balance_ts, ca_ts, u_total_ts, balance


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
    "nhits_params, fit_params, dependent_variables",
    [
        ( # самый базовый кейс с дефолтными значениями параметров
            NhitsParams(),
            FitParams(),
            Timeseries()
        ),
        ( # месячные данные
            NhitsParams(),
            FitParams(
                train_boundary=datetime(2020, 12, 31),
                val_boundary=datetime(2024, 1,31)
            ),
            balance_ts()
        ),
        ( # квартальные данные
            NhitsParams(),
            FitParams(
                train_boundary=datetime(2019, 6, 30),
                val_boundary=datetime(2021, 6, 30),
                forecast_horizon=3,
            ),
            ca_ts()
        ),
        ( # годовые данные
            NhitsParams(),
            FitParams(
                train_boundary=datetime(2012, 12, 31),
                val_boundary=datetime(2021, 12, 31),
                forecast_horizon=3,
            ),
            u_total_ts()
        )
    ]
)
def test_nhits_fit_without_exog(
    nhits_params,
    fit_params,
    dependent_variables,
    client
):
    data = dict(
        dependent_variables=process_variable(dependent_variables),
        explanatory_variables=None,
        hyperparameters=nhits_params.model_dump(),
        fit_params=process_fit_params(fit_params),
    )
    result = client.post(
        url='/api/v1/building_model/nhits/fit',
        json=data
    )

    received_data = result.json()
    assert result.status_code == 200, received_data

    validate_no_exog_result(received_data, dependent_variables, data, fit_params)


# Параметры для перебора
FORECAST_HORIZONS = [1, 6, 12, 24, 36]
TEST_SIZES = [0, 12, 24, 36]
VAL_SIZES = [0, 12, 24, 36]


total_points = 30
FORECAST_HORIZONS_2 = [i for i in range(total_points)]
TEST_SIZES_2 = [i for i in range(total_points)]
VAL_SIZES_2 = [i for i in range(total_points)]


def generate_valid_combinations(f, t, v):
    """Генерирует допустимые комбинации параметров"""
    combinations = []
    for h in f:
        for test_size in t:
            for val_size in v:
                train_size = total_points - val_size - test_size

                # Проверка ограничения 3
                if 4 * (h + test_size) > train_size:
                    continue

                # Проверка ограничения 1
                if 0 < val_size < h + test_size:
                    continue

                # Проверка минимального размера train
                if train_size < 10:  # Минимум 10 точек для обучения
                    continue

                if h + test_size == 0:
                    continue

                combinations.append((h, test_size, val_size))

    return combinations

VALID_COMBINATIONS = generate_valid_combinations(
    FORECAST_HORIZONS, TEST_SIZES, VAL_SIZES
)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize('n_stacks', [1, 2, 3])
@pytest.mark.parametrize('activation', [ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid])
@pytest.mark.parametrize('scaler_type', [ScalerType.Standard, ScalerType.Robust, ScalerType.Identity])
@pytest.mark.parametrize('early_stop_patience_steps', [-1])
@pytest.mark.parametrize('pooling_mode', [PoolingMode.AvgPool1d])
@pytest.mark.parametrize('interpolation_mode', [InterpMode.Linear,])
@pytest.mark.parametrize('loss', [LossEnum.MAE])
@pytest.mark.parametrize('valid_loss', [LossEnum.MSE])
@pytest.mark.parametrize("learning_rate", [1e-4,])
@pytest.mark.parametrize(
    "h, test_size, val_size",
    VALID_COMBINATIONS,
)
def test_nhits_fit_without_exog_grid_params(
        n_stacks: int,
        activation: ActivationType,
        scaler_type: ScalerType,
        early_stop_patience_steps,
        pooling_mode,
        interpolation_mode,
        loss,
        valid_loss,
        learning_rate,
        h: int,
        test_size: int,
        val_size: int,
        client,
        balance,
):
    # 1. Подготовка временного ряда (401 месяц)
    total_points = 401
    dates = pd.date_range(start="1992-01-31", periods=total_points, freq="ME")

    # 2. Рассчет границ выборок
    train_size = total_points - val_size - test_size
    train_end_idx = train_size - 1
    val_end_idx = train_end_idx + val_size

    # 3. Установка границ дат
    train_boundary_date = dates[train_end_idx]
    val_boundary_date = dates[val_end_idx] if val_size > 0 else train_boundary_date

    # 4. Настройка параметров
    fit_params = FitParams(
        train_boundary=train_boundary_date,
        val_boundary=val_boundary_date,
        forecast_horizon=h
    )

    # Автоподбор параметров под размеры выборок
    n_blocks = [1] * n_stacks
    n_pool_kernel_size = [2] * n_stacks

    nhits_params = NhitsParams(
        n_stacks=n_stacks,
        n_blocks=n_blocks,
        n_pool_kernel_size=n_pool_kernel_size,
        activation=activation,
        scaler_type=scaler_type,
        early_stop_patience_steps=early_stop_patience_steps,
        pooling_mode=pooling_mode,
        interpolation_mode=interpolation_mode,
        loss=loss,
        valid_loss=valid_loss,
        # Фиксируем остальные параметры
        max_steps=100,
        val_check_steps=random.randint(0, 1000),
        learning_rate=learning_rate,
    )

    data = dict(
        dependent_variables=process_variable(balance),
        explanatory_variables=None,
        hyperparameters=nhits_params.model_dump(),
        fit_params=process_fit_params(fit_params),
    )
    result = client.post(
        url='/api/v1/building_model/nhits/fit',
        json=data
    )

    received_data = result.json()
    assert result.status_code == 200, received_data

    # Адаптированная проверка для случаев без теста
    forecasts = received_data['forecasts']
    test_predict = forecasts['test_predict']

    assert received_data['model_metrics']['train_metrics'], "Train metrics missing"

    # Проверяем test_metrics только если есть тестовые данные
    if test_predict:
        assert received_data['model_metrics']['test_metrics'], "Test metrics missing"
        validate_no_exog_result(received_data, balance, data, fit_params)
    else:
        assert received_data['model_metrics']['test_metrics'] is None, "Test metrics should be empty"
        validate_empty_test_data(received_data, balance, data, fit_params)

    assert received_data['weight_path'], "Weight path missing"


VALID_COMBINATIONS_EXTENDED = generate_valid_combinations(
    FORECAST_HORIZONS_2, TEST_SIZES_2, VAL_SIZES_2
)

@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "h, test_size, val_size",
    VALID_COMBINATIONS_EXTENDED,
)
def test_nhits_fit_grid_params(
    h: int,
    test_size: int,
    val_size: int,
    client,
    balance,
):
    balance = Timeseries(
        data_frequency=balance.data_frequency,
        dates=balance.dates[:total_points],
        values=balance.values[:total_points],
        name="balance",
    )
    # 1. Подготовка временного ряда (401 месяц)
    dates = balance.dates
    total_size = len(dates)
    assert total_points == total_size
    # 2. Рассчет границ выборок
    train_size = total_size - val_size - test_size
    train_end_idx = train_size - 1
    val_end_idx = train_end_idx + val_size

    # 3. Установка границ дат
    train_boundary_date = dates[train_end_idx]
    val_boundary_date = dates[val_end_idx]

    # 4. Настройка параметров
    fit_params = FitParams(
        train_boundary=train_boundary_date,
        val_boundary=val_boundary_date,
        forecast_horizon=h
    )

    nhits_params = NhitsParams(
        early_stop_patience_steps=-1,
        n_stacks=2,
        n_blocks=[1, 1],
        n_pool_kernel_size=[2, 1],
        pooling_mode=PoolingMode.AvgPool1d,
        interpolation_mode=InterpMode.Linear,
        loss=LossEnum.MAE,
        valid_loss=LossEnum.MAE,
        activation=ActivationType.ReLU,
        max_steps=20,
        val_check_steps=50,
        learning_rate=1e-3,
        scaler_type=ScalerType.Identity
    )

    # 6. Отправка запроса
    data = dict(
        dependent_variables=process_variable(balance),
        explanatory_variables=None,
        hyperparameters=nhits_params.model_dump(),
        fit_params=process_fit_params(fit_params),
    )
    result = client.post(
        url='/api/v1/building_model/nhits/fit',
        json=data
    )

    received_data = result.json()
    assert result.status_code == 200, received_data

    # Адаптированная проверка для случаев без теста
    forecasts = received_data['forecasts']
    test_predict = forecasts['test_predict']

    assert received_data['model_metrics']['train_metrics'], "Train metrics missing"

    # Проверяем test_metrics только если есть тестовые данные
    if test_predict:
        assert received_data['model_metrics']['test_metrics'], "Test metrics missing"
        validate_no_exog_result(received_data, balance, data, fit_params)
    else:
        assert received_data['model_metrics']['test_metrics'] is None, "Test metrics should be empty"
        validate_empty_test_data(received_data, balance, data, fit_params)

    assert received_data['weight_path'], "Weight path missing"
