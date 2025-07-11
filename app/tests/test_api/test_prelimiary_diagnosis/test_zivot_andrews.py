import random

import numpy as np
import pytest

from src.core.application.preliminary_diagnosis.schemas.zivot_andrews import RegressionEnum, AutoLagEnum
from src.core.domain import Timeseries
from tests.conftest import client, balance_ts

target = balance_ts()
reduced_target = Timeseries(
    name="reduced_target",
    values=target.values[:10],
    dates=target.dates[:10],
    data_frequency=target.data_frequency,
)


def process_variable(ts: Timeseries) -> dict:
    return {
        "name": ts.name,
        "values": ts.values,
        "dates": [date.strftime("%Y-%m-%d") for date in ts.dates],
        "data_frequency": ts.data_frequency,
    }

@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("lags", [None, 1] + [i for i in range(2, len(reduced_target.values) + 3)])
@pytest.mark.parametrize("trim", [0.0, 0.33] + np.linspace(start=0.0, stop=0.33, num=20).tolist())
@pytest.mark.parametrize("max_lag", [None, 1] + [i for i in range(2, len(reduced_target.values) + 3)])
@pytest.mark.parametrize("regression", [
    RegressionEnum.ConstantOnly, RegressionEnum.ConstantAndTrend, RegressionEnum.TrendOnly
])
@pytest.mark.parametrize("autolag", [AutoLagEnum.t_stat, AutoLagEnum.AIC, AutoLagEnum.BIC])
def test_zivot_andrews_short(
        lags,
        trim,
        max_lag,
        regression,
        autolag,
        client,
):
    ts = process_variable(reduced_target)
    result = client.post(
        url='/api/v1/stationary_testing/zivot_andrews',
        json={
            "ts": ts,
            "lags": lags,
            "trim": trim,
            "max_lags": max_lag,
            "regression": regression,
            "autolag": autolag
        }
    )
    data = result.json()
    if (
        result.status_code == 400 and
        (
            "observations are needed to run an ADF" in data.get('detail', "") or
            "The maximum lag you are considering" in data.get('detail', "") or
            "maxlag should be < nobs" in data.get('detail', "") or
            "The regressor matrix is singular." in data.get('detail', "") or
            "The number of observations is too small to use the Zivot-Andrews" in data.get('detail', "")
        )
    ):
        assert True
    else:
        assert result.status_code == 200