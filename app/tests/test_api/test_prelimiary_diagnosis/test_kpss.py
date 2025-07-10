import pytest

from src.core.application.preliminary_diagnosis.schemas.kpss import RegressionEnum
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
@pytest.mark.parametrize('regression', [RegressionEnum.ConstantOnly, RegressionEnum.ConstantAndTrend])
@pytest.mark.parametrize("nlags", [i for i in range(len(reduced_target.dates) + 10)])
def test_kpss_short(
        regression,
        nlags,
        client,
):
    ts = process_variable(reduced_target)
    result = client.post(
        url='/api/v1/stationary_testing/kpss',
        json={
            "ts": ts,
            "nlags": nlags,
            "regression": regression
        }
    )
    data = result.json()
    if nlags < len(reduced_target.dates):
        assert result.status_code == 200, data
    else:
        assert result.status_code == 400, data


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize('regression', [RegressionEnum.ConstantOnly, RegressionEnum.ConstantAndTrend])
@pytest.mark.parametrize("nlags", [i for i in range(len(target.dates) + 10)])
def test_kpss_long(
        regression,
        nlags,
        client,
):
    ts = process_variable(target)
    result = client.post(
        url='/api/v1/stationary_testing/kpss',
        json={
            "ts": ts,
            "nlags": nlags,
            "regression": regression
        }
    )
    data = result.json()
    if nlags < len(target.dates):
        assert result.status_code == 200, data
    else:
        assert result.status_code == 400, data
