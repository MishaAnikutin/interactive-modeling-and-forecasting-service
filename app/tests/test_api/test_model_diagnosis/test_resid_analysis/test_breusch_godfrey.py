import pytest

from tests.conftest import balance_ts
from tests.test_api.utils import process_variable
from .fixtures import forecasts_lstm_base, forecasts_lstm_exog

def test_dg_without_exog(client, forecasts_lstm_base):
    dg_json = {
        "data": {
            "forecasts": forecasts_lstm_base["forecasts"],
            "target": process_variable(balance_ts()),
            "exog": None
        }
    }

    dg_result = client.post(
        url='/api/v1/model_diagnosis/resid_analysis/breusch_godfrey',
        json=dg_json
    )

    data_dg = dg_result.json()
    assert dg_result.status_code == 200, data_dg

@pytest.mark.parametrize("nlags", [None, ] + list(range(1, 100, 29)))
def test_dg_with_exog(
        nlags,
        balance,
        ipp_eu_ts,
        client,
        forecasts_lstm_exog
):
    dg_json = {
        "data": {
            "forecasts": forecasts_lstm_exog["forecasts"],
            "target": process_variable(balance),
            "exog": [process_variable(ipp_eu_ts)]
        },
        "nlags": nlags
    }

    dg_result = client.post(
        url='/api/v1/model_diagnosis/resid_analysis/breusch_godfrey',
        json=dg_json
    )

    data_dg = dg_result.json()
    assert dg_result.status_code == 200, data_dg