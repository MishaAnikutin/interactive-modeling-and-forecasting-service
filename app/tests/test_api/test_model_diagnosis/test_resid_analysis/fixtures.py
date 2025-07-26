from datetime import date

from pytest import fixture

from src.core.application.building_model.schemas.lstm import LstmParams
from src.core.domain import FitParams
from tests.conftest import balance_ts
from tests.test_api.test_building_model.validators import process_fit_params
from tests.test_api.utils import process_variable

@fixture(scope="function")
def forecasts_lstm_base(client):
    fit_params = FitParams(
        train_boundary=date(2020, 12, 31),
        val_boundary=date(2024, 1, 31)
    )
    data = dict(
        dependent_variables=process_variable(balance_ts()),
        explanatory_variables=None,
        hyperparameters=LstmParams().model_dump(),
        fit_params=process_fit_params(fit_params),
    )
    result = client.post(
        url='/api/v1/building_model/lstm/fit',
        json=data
    )

    received_data = result.json()
    assert result.status_code == 200, received_data
    return received_data