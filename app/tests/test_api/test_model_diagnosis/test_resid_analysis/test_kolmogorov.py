from tests.conftest import balance_ts
from tests.test_api.utils import process_variable
from .fixtures import forecasts_lstm_base

def test_kstest(client, forecasts_lstm_base):
    kstest_json = {
        "data": {
            "forecasts": forecasts_lstm_base["forecasts"],
            "target": process_variable(balance_ts()),
            "exog": None
        }
    }

    kstest_result = client.post(
        url='/api/v1/model_diagnosis/resid_analysis/kstest_normal',
        json=kstest_json
    )

    data_kstest = kstest_result.json()
    assert kstest_result.status_code == 200, data_kstest
