import pytest
from statsmodels.tsa.tsatools import freq_to_period

from src.core.application.generating_series.schemas.naive_decomposition import NaiveDecompositionParams, ModelEnum
from tests.test_api.utils import process_variable


def test_naive_base(client, u_total):
    ts = process_variable(u_total)
    params = NaiveDecompositionParams(filt=None).model_dump()

    data = {
        "ts": ts,
        "params": params,
    }

    result = client.post(
        url='/api/v1/generating_series/seasonal_decomposition/naive',
        json=data
    )

    data = result.json()
    assert result.status_code == 200, data

    observed = data["observed"]
    trend = data["trend"]
    seasonal = data["seasonal"]
    resid = data["resid"]

    for calculated_ts in [observed, trend, seasonal, resid]:
        assert len(calculated_ts['dates']) == len(ts['dates'])
        assert calculated_ts['dates'] == ts['dates']
        assert calculated_ts['data_frequency'] == ts['data_frequency']


@pytest.mark.slow
@pytest.mark.parametrize("model", [ModelEnum.additive, ModelEnum.multiplicative])
@pytest.mark.parametrize("period", [None, 1,]  + list(range(1, 100, 3)))
@pytest.mark.parametrize("two_sided", [True, False])
@pytest.mark.parametrize("filt", [None, [1, 2, 3, 4], [-1, 0, 1, 2, 10], list(range(32))])
@pytest.mark.parametrize("extrapolate_trend", [None, 1] + list(range(1, 50, 3)))
def test_naive_decomp_grid_params(
        model,
        period,
        two_sided,
        filt,
        extrapolate_trend,
        client,
        u_total
):
    ts = process_variable(u_total)
    params = NaiveDecompositionParams(
        model=model,
        period=period,
        two_sided=two_sided,
        filt=filt,
        extrapolate_trend=extrapolate_trend,
    ).model_dump()

    data = {
        "ts": ts,
        "params": params,
    }

    result = client.post(
        url='/api/v1/generating_series/seasonal_decomposition/naive',
        json=data
    )

    data = result.json()
    pperiod = period if period is not None else freq_to_period(u_total.data_frequency)
    if len(ts['dates']) < 2 * pperiod:
        assert result.status_code == 422, data
        return
    assert result.status_code == 200, data

    observed = data["observed"]
    trend = data["trend"]
    seasonal = data["seasonal"]
    resid = data["resid"]

    for calculated_ts in [observed, trend, seasonal, resid]:
        assert len(calculated_ts['dates']) == len(ts['dates'])
        assert calculated_ts['dates'] == ts['dates']
        assert calculated_ts['data_frequency'] == ts['data_frequency']
