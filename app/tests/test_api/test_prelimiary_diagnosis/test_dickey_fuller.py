import numpy as np

from src.core.application.preliminary_diagnosis.schemas.dickey_fuller import AutoLagEnum, RegressionEnum
from src.core.domain import Timeseries
from tests.conftest import client, balance_ts
import pytest

target = balance_ts()

regression_possible = [
    RegressionEnum.ConstantOnly,
    RegressionEnum.ConstantAndTrend,
    RegressionEnum.ConstantLinearAndQuadraticTrend,
    RegressionEnum.NoConstantNoTrend
]

def get_valid_combinations(total_size: int):
    max_lags_possible = [None, 1] + list(range(2, total_size + 10))
    result = []
    for regression in regression_possible:
        ntrend = len(regression) if regression != "n" else 0
        for max_lags in max_lags_possible:
            if max_lags is None:
                # from Greene referencing Schwert 1989
                maxlag = int(np.ceil(12.0 * np.power(total_size / 100.0, 1 / 4.0)))
                # -1 for the diff
                maxlag = min(total_size // 2 - ntrend - 1, maxlag)
                if maxlag < 0:
                    continue
            elif max_lags > total_size // 2 - ntrend - 1:
                continue
            result.append((max_lags, regression))
    return result

def process_variable(ts: Timeseries) -> dict:
    return {
        "name": ts.name,
        "values": ts.values,
        "dates": [date.strftime("%Y-%m-%d") for date in ts.dates],
        "data_frequency": ts.data_frequency,
    }

@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize('autolag', [AutoLagEnum.AIC, AutoLagEnum.BIC, AutoLagEnum.t_stat, None])
@pytest.mark.parametrize(
    "max_lags, regression",
    get_valid_combinations(len(target.dates)),
)
def test_adf_base(
        autolag,
        max_lags,
        regression,
        client,
):
    ts = process_variable(target)
    result = client.post(
        url='/api/v1/stationary_testing/dickey_fuller',
        json={
            "ts": ts,
            "max_lags": max_lags,
            "autolag": autolag,
            "regression": regression
        }
    )
    data = result.json()
    assert result.status_code == 200, data


reduced_target = Timeseries(
    name="reduced_target",
    values=target.values[:10],
    dates=target.dates[:10],
    data_frequency=target.data_frequency,
)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize('autolag', [AutoLagEnum.AIC, AutoLagEnum.BIC, AutoLagEnum.t_stat, None])
@pytest.mark.parametrize(
    "max_lags, regression",
    get_valid_combinations(len(reduced_target.dates)),
)
def test_adf_short(
        autolag,
        max_lags,
        regression,
        client,
):
    ts = process_variable(reduced_target)
    result = client.post(
        url='/api/v1/stationary_testing/dickey_fuller',
        json={
            "ts": ts,
            "max_lags": max_lags,
            "autolag": autolag,
            "regression": regression
        }
    )
    data = result.json()
    assert result.status_code == 200, data


empty_target = Timeseries(
    name='empty',
    values=[],
    dates=[],
    data_frequency='ME'
)

@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_adf_empty_ts(client):
    ts = process_variable(empty_target)
    result = client.post(
        url='/api/v1/stationary_testing/dickey_fuller',
        json={
            "ts": ts,
            "max_lags": None,
            "autolag": None,
            "regression": 'c'
        }
    )
    data = result.json()
    assert result.status_code == 422, data
