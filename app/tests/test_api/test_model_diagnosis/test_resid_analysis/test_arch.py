from tests.conftest import balance_ts
from tests.test_api.utils import process_variable
from .fixtures import forecasts_lstm_base
import pytest

@pytest.mark.slow
@pytest.mark.parametrize("nlags", [None, 0] + list(range(1, 100, 29)))
@pytest.mark.parametrize("period", [None, 2] + list(range(3, 400, 52)))
@pytest.mark.parametrize("ddof", list(range(0, 100, 18)))
@pytest.mark.parametrize("cov_type", ["nonrobust", "HC2"])
def test_arch(
        nlags,
        period,
        ddof,
        cov_type,
        client,
        forecasts_lstm_base
):
    arch_json = {
        "data": {
            "forecasts": forecasts_lstm_base["forecasts"],
            "ts": process_variable(balance_ts()),
        },
    }

    arch_result = client.post(
        url='/api/v1/model_diagnosis/resid_analysis/arch',
        json=arch_json
    )

    data_arch = arch_result.json()
    assert arch_result.status_code == 200, data_arch