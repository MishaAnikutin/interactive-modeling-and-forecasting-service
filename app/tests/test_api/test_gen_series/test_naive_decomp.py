import pytest

from src.core.application.generating_series.schemas.naive_decomposition import NaiveDecompositionParams
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