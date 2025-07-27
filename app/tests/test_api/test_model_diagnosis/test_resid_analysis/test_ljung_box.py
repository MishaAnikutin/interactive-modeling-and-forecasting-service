from src.core.application.model_diagnosis.schemas.common import ResidAnalysisData
from src.core.application.model_diagnosis.schemas.ljung_box import LjungBoxRequest
from tests.conftest import balance_ts
from tests.test_api.utils import process_variable
from .fixtures import forecasts_lstm_base
import pytest

@pytest.mark.slow
@pytest.mark.parametrize("lags", [None, 1] + [[1, 2, 4], [24, 5]])
@pytest.mark.parametrize("period", [None, 2] + list(range(3, 400, 52)))
@pytest.mark.parametrize("model_df", list(range(0, 100, 18)))
@pytest.mark.parametrize("auto_lag", [False, True])
def test_ljung_box(
        lags,
        period,
        model_df,
        auto_lag,
        client,
        forecasts_lstm_base
):
    data = dict(
        data=dict(
            forecasts=forecasts_lstm_base["forecasts"],
            target=process_variable(balance_ts()),
            exog=None
        ),
        lags=lags,
        period=period,
        model_df=model_df,
        auto_lag=auto_lag,
    )

    ljung_box_result = client.post(
        url='/api/v1/model_diagnosis/resid_analysis/ljung_box',
        json=data
    )

    data_ljung_box = ljung_box_result.json()
    assert ljung_box_result.status_code == 200, data_ljung_box