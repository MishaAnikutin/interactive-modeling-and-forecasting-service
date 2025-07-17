from src.core.application.generating_series.schemas.stl_decomposition import STLParams
from src.core.domain import Timeseries
from tests.test_api.utils import process_variable, delete_timestamp


def test_stl_base(
    client,
    balance
):
    ts = process_variable(Timeseries())
    params = STLParams().model_dump()

    data = {
        "ts": ts,
        "params": params,
    }

    result = client.post(
        url='/api/v1/generating_series/seasonal_decomposition/stl',
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
        assert delete_timestamp(calculated_ts['dates']) == ts['dates']
        assert calculated_ts['data_frequency'] == ts['data_frequency']

