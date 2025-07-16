from datetime import datetime
import pytest

from src.core.application.building_model.schemas.lstm import LstmParams
from src.core.domain import FitParams, Timeseries
from tests.common.params_permutations import VALID_COMBINATIONS_EXTENDED_exog, aligned_size
from tests.conftest import client, balance_ts, ca_ts, u_total_ts, balance, ipp_eu_ts
from tests.test_api.test_building_model.validators import process_variable, process_fit_params, validate_no_exog_result, \
    validate_only_train_data, validate_empty_val_data, validate_empty_test_data


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


@pytest.mark.parametrize(
    "h, test_size, val_size",
    VALID_COMBINATIONS_EXTENDED_exog,
)
def test_lstm_fit_with_exog(
    h: int,
    test_size: int,
    val_size: int,
    client,
    balance,
    ipp_eu_ts,
):
    ipp_eu_ts_reduced = Timeseries(
        name=ipp_eu_ts.name,
        data_frequency=ipp_eu_ts.data_frequency,
        dates=ipp_eu_ts.dates[:30],
        values=ipp_eu_ts.values[:30],
    )
    min_date = max(balance.dates[0], ipp_eu_ts_reduced.dates[0])
    max_date = min(balance.dates[-1], ipp_eu_ts_reduced.dates[-1])
    min_date_index = balance.dates.index(min_date)
    max_date_index = balance.dates.index(max_date)
    dates = balance.dates[min_date_index:max_date_index]
    total_size = len(dates)
    aligned_balance = Timeseries(
        name="aligned_balance",
        data_frequency=balance.data_frequency,
        dates=dates,
        values=balance.values[min_date_index:max_date_index],
    )
    assert aligned_size == total_size
    assert len(aligned_balance.dates) == len(aligned_balance.values)
    # 2. Рассчет границ выборок
    train_size = total_size - val_size - test_size
    train_end_idx = train_size - 1
    val_end_idx = train_end_idx + val_size

    # 3. Установка границ дат
    train_boundary_date = dates[train_end_idx]
    val_boundary_date = dates[val_end_idx]

    fit_params = FitParams(
        train_boundary=train_boundary_date,
        val_boundary=val_boundary_date,
        forecast_horizon=h
    )

    lstm_params = LstmParams()

    # 6. Отправка запроса
    data = dict(
        dependent_variables=process_variable(aligned_balance),
        explanatory_variables=[process_variable(ipp_eu_ts_reduced)],
        hyperparameters=lstm_params.model_dump(),
        fit_params=process_fit_params(fit_params),
    )
    result = client.post(
        url='/api/v1/building_model/lstm/fit',
        json=data
    )

    received_data = result.json()
    assert result.status_code == 200, received_data

    assert received_data['model_metrics']['train_metrics'], "Train metrics missing"

    if test_size and val_size:
        assert received_data['model_metrics']['test_metrics'], "Test metrics missing"
        assert received_data['model_metrics']['val_metrics'], "Validation metrics missing"
        validate_no_exog_result(received_data, aligned_balance, data, fit_params)
    elif (not test_size) and val_size:
        assert received_data['model_metrics']['test_metrics'] is None, "Test metrics should be empty"
        assert received_data['model_metrics']['val_metrics'], "Validation metrics missing"
        validate_empty_test_data(received_data, aligned_balance, data, fit_params)
    elif test_size and (not val_size):
        assert received_data['model_metrics']['test_metrics'], "Test metrics missing"
        assert received_data['model_metrics']['val_metrics'] is None, "Validation metrics should be empty"
        validate_empty_val_data(received_data, aligned_balance, data, fit_params)
    elif (not val_size) and (not test_size):
        assert received_data['model_metrics']['test_metrics'] is None, "Test metrics should be empty"
        assert received_data['model_metrics']['val_metrics'] is None, "Validation metrics should be empty"
        validate_only_train_data(received_data, aligned_balance, data, fit_params)

    assert received_data['weight_path'], "Weight path missing"

@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("input_size", [1, 10, 5000])
@pytest.mark.parametrize("inference_input_size", [1, 10, 5000])
@pytest.mark.parametrize("h_train", [1, 10, 5000])
@pytest.mark.parametrize("encoder_n_layers", [2, 4])
@pytest.mark.parametrize("encoder_hidden_size", [2])
@pytest.mark.parametrize("decoder_hidden_size", [10, 128])
@pytest.mark.parametrize("decoder_layers", [0, 2])
@pytest.mark.parametrize("recurrent", [True, False])
@pytest.mark.parametrize(
    "h, test_size, val_size",
    [(3, 1, 7)],
)
def test_lstm_fit_exog_grid_params(
        input_size: int,
        inference_input_size,
        h_train: int,
        encoder_n_layers: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        decoder_layers: int,
        recurrent: bool,
        h: int,
        test_size: int,
        val_size: int,
        client,
        balance,
        ipp_eu_ts,
):
    ipp_eu_ts_reduced = Timeseries(
        name=ipp_eu_ts.name,
        data_frequency=ipp_eu_ts.data_frequency,
        dates=ipp_eu_ts.dates[:30],
        values=ipp_eu_ts.values[:30],
    )
    min_date = max(balance.dates[0], ipp_eu_ts_reduced.dates[0])
    max_date = min(balance.dates[-1], ipp_eu_ts_reduced.dates[-1])
    min_date_index = balance.dates.index(min_date)
    max_date_index = balance.dates.index(max_date)
    dates = balance.dates[min_date_index:max_date_index]
    total_size = len(dates)
    aligned_balance = Timeseries(
        name="aligned_balance",
        data_frequency=balance.data_frequency,
        dates=dates,
        values=balance.values[min_date_index:max_date_index],
    )
    assert aligned_size == total_size
    assert len(aligned_balance.dates) == len(aligned_balance.values)
    # 2. Рассчет границ выборок
    train_size = total_size - val_size - test_size
    train_end_idx = train_size - 1
    val_end_idx = train_end_idx + val_size

    # 3. Установка границ дат
    train_boundary_date = dates[train_end_idx]
    val_boundary_date = dates[val_end_idx]

    fit_params = FitParams(
        train_boundary=train_boundary_date,
        val_boundary=val_boundary_date,
        forecast_horizon=h
    )

    lstm_params = LstmParams(
        input_size=input_size,
        inference_input_size=inference_input_size,
        h_train=h_train,
        encoder_n_layers=encoder_n_layers,
        encoder_hidden_size=encoder_hidden_size,
        decoder_hidden_size=decoder_hidden_size,
        decoder_layers=decoder_layers,
        recurrent=recurrent,
    )

    # 6. Отправка запроса
    data = dict(
        dependent_variables=process_variable(aligned_balance),
        explanatory_variables=[process_variable(ipp_eu_ts_reduced)],
        hyperparameters=lstm_params.model_dump(),
        fit_params=process_fit_params(fit_params),
    )
    result = client.post(
        url='/api/v1/building_model/lstm/fit',
        json=data
    )

    received_data = result.json()
    if lstm_params.input_size + fit_params.forecast_horizon + test_size > total_size - test_size - val_size:
        assert result.status_code == 400, received_data
        return
    elif recurrent:
        if lstm_params.input_size + h_train + test_size > total_size - test_size - val_size:
            assert result.status_code == 400, received_data
            return
    assert result.status_code == 200, received_data

    assert received_data['model_metrics']['train_metrics'], "Train metrics missing"

    assert received_data['model_metrics']['test_metrics'], "Test metrics missing"
    assert received_data['model_metrics']['val_metrics'], "Validation metrics missing"
    validate_no_exog_result(received_data, aligned_balance, data, fit_params)

    assert received_data['weight_path'], "Weight path missing"