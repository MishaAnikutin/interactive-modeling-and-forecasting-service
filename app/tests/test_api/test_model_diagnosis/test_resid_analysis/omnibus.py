from datetime import date

from src.core.application.building_model.schemas.lstm import LstmParams
from src.core.domain import FitParams
from tests.conftest import balance_ts
from tests.test_api.test_building_model.validators import process_fit_params
from tests.test_api.utils import process_variable


def test_omnibus(client):
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

    omni_json = {
        "data": {
            "forecasts": received_data["forecasts"],
            "ts": process_variable(balance_ts()),
        }
    }

    omni_result = client.post(
        url='/api/v1/model_diagnosis/resid_analysis/omnibus',
        json=omni_json
    )

    data_omni = omni_result.json()
    assert omni_result.status_code == 200, data_omni

    print(data_omni)