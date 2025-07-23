import pytest

from src.core.application.preliminary_diagnosis.schemas.df_gls import TrendEnum, MethodEnum
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

trend_values = [TrendEnum.ConstantAndTrend, TrendEnum.ConstantOnly]

def get_valid_combinations(total: int):
    lags_values = [None, 1] + [i for i in range(2, total + 10)]
    result = []
    for trend in trend_values:
        trend_order = len(trend)
        for lag in lags_values:
            lag_len = 0 if lag is None else lag
            required = 3 + trend_order + lag_len
            if total < required:
                continue
            result.append((trend, lag))
    return result


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    'trend, lags',
    get_valid_combinations(len(reduced_target.dates))
)
@pytest.mark.parametrize('method', [MethodEnum.AIC, MethodEnum.BIC, MethodEnum.t_stat])
@pytest.mark.parametrize("max_lags", [None, 1] + [i for i in range(2, len(reduced_target.values) + 10)])
def test_df_gls_short(
        trend,
        method,
        lags,
        max_lags,
        client,
):
    ts = process_variable(reduced_target)
    result = client.post(
        url='/api/v1/preliminary_diagnosis/stationary_testing/df_gls',
        json={
            "ts": ts,
            "lags": lags,
            "trend": trend,
            "method": method,
            "max_lags": max_lags
        }
    )
    data = result.json()
    if (
        result.status_code == 400 and
        (
            "observations are needed to run an ADF" in data.get('detail', "") or
            "The maximum lag you are considering" in data.get('detail', "") or
            "maxlag should be < nobs" in data.get('detail', "")
        )
    ):
        assert True
    else:
        assert result.status_code == 200
