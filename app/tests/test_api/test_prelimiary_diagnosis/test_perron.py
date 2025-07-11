import pytest

from src.core.application.preliminary_diagnosis.schemas.phillips_perron import TestType, TrendEnum
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
@pytest.mark.parametrize('trend', [TrendEnum.ConstantAndTrend, TrendEnum.ConstantOnly, TrendEnum.NoConstantNoTrend])
@pytest.mark.parametrize('test_type', [TestType.tau, TestType.rho])
@pytest.mark.parametrize("lags", [None, 1] + [i for i in range(2, len(reduced_target.values) + 10)])
def test_perron_short(
        trend,
        test_type,
        lags,
        client,
):
    ts = process_variable(reduced_target)
    result = client.post(
        url='/api/v1/stationary_testing/phillips_perron',
        json={
            "ts": ts,
            "lags": lags,
            "trend": trend,
            "test_type": test_type,
        }
    )
    data = result.json()
    if ((lags is not None) and (lags <= len(reduced_target.values) - 1)) or lags is None:
        assert result.status_code == 200, data
    else:
        assert result.status_code == 422, data


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize('trend', [TrendEnum.ConstantAndTrend, TrendEnum.ConstantOnly, TrendEnum.NoConstantNoTrend])
@pytest.mark.parametrize('test_type', [TestType.tau, TestType.rho])
@pytest.mark.parametrize("lags", [None, 1] + [i for i in range(2, len(target.values) + 10)])
def test_perron_long(
        trend,
        test_type,
        lags,
        client,
):
    ts = process_variable(reduced_target)
    result = client.post(
        url='/api/v1/stationary_testing/phillips_perron',
        json={
            "ts": ts,
            "lags": lags,
            "trend": trend,
            "test_type": test_type,
        }
    )
    data = result.json()
    if ((lags is not None) and (lags <= len(reduced_target.values) - 1)) or lags is None:
        assert result.status_code == 200, data
    else:
        assert result.status_code == 422, data