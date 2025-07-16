from datetime import datetime
import pytest

from src.core.application.building_model.schemas.lstm import LstmParams
from src.core.domain import FitParams, Timeseries
from tests.conftest import client, balance_ts, ca_ts, u_total_ts, balance, ipp_eu_ts
from tests.test_api.test_building_model.validators import process_variable, process_fit_params, validate_no_exog_result


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