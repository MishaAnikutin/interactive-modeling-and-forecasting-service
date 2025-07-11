import pytest

from src.core.domain import Timeseries
from tests.conftest import balance_ts, client

target = balance_ts()
reduced_target = Timeseries(
    name="reduced_target",
    values=target.values[:10],
    dates=target.dates[:10],
    data_frequency=target.data_frequency,
)

reduced_target_2 = Timeseries(
    name="reduced_target",
    values=target.values[:25],
    dates=target.dates[:25],
    data_frequency=target.data_frequency,
)

def process_variable(ts: Timeseries) -> dict:
    return {
        "name": ts.name,
        "values": ts.values,
        "dates": [date.strftime("%Y-%m-%d") for date in ts.dates],
        "data_frequency": ts.data_frequency,
    }

@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("ts", [target, reduced_target, reduced_target_2])
def test_range_unit_root(
    ts,
    client,
):
    ts_processed = process_variable(ts)
    result = client.post(
        url='/api/v1/stationary_testing/range_unit_root',
        json={"ts": ts_processed}
    )
    data = result.json()
    if len(ts.values) < 25:
        assert result.status_code == 422, data
    else:
        assert result.status_code == 200, data
