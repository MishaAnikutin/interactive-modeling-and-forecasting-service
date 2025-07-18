from datetime import datetime
from typing import Optional

import pandas as pd
import pytest

from src.core.application.generating_series.schemas.stl_decomposition import STLParams
from src.core.domain import Timeseries, DataFrequency
from tests.test_api.utils import process_variable, delete_timestamp


def test_stl_base(client, u_total):
    ts = process_variable(u_total)
    params = STLParams(trend=5, period=2).model_dump()

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


@pytest.mark.slow
@pytest.mark.parametrize("seasonal", [3, 7, 21, 33])
@pytest.mark.parametrize("trend", [None, 13, 15, 35])
@pytest.mark.parametrize("low_pass", [None, 13, 35])
@pytest.mark.parametrize("seasonal_deg", ["0", "1"])
@pytest.mark.parametrize("trend_deg", ["0", "1"])
@pytest.mark.parametrize("low_pass_deg", ["0", "1"])
@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("seasonal_jump", [1, 10, 35])
@pytest.mark.parametrize("trend_jump", [1, 10, 35])
@pytest.mark.parametrize("low_pass_jump", [1, 10, 35])
def test_stl_grid_params(
    seasonal: int,
    trend: Optional[int],
    low_pass: Optional[int],
    seasonal_deg: str,
    trend_deg: str,
    low_pass_deg: str,
    robust: bool,
    seasonal_jump: int,
    trend_jump: int,
    low_pass_jump: int,
    client,
    balance
):
    ts = process_variable(balance)
    params = STLParams(
        seasonal=seasonal,
        trend=trend,
        low_pass=low_pass,
        seasonal_deg=seasonal_deg,
        trend_deg=trend_deg,
        low_pass_deg=low_pass_deg,
        robust=robust,
        seasonal_jump=seasonal_jump,
        trend_jump=trend_jump,
        low_pass_jump=low_pass_jump,
    ).model_dump()

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


def gen_dates_day() -> list[datetime]:
    """Возвращает n дат типа «month ending», начиная с 31-01-2023."""
    n = 100
    dates_idx = pd.date_range(start="2023-01-31", periods=n, freq="D")
    return [d.to_pydatetime() for d in dates_idx]

@pytest.mark.slow
@pytest.mark.parametrize("period", [None, 2, 7, 21, 33, 365, 377])
@pytest.mark.parametrize("seasonal", [3, 7, 21, 33, 365])
@pytest.mark.parametrize("trend", [None, 3, 13, 15, 35, 365, 377])
@pytest.mark.parametrize("low_pass", [None, 3, 13, 35, 365, 377])
def test_stl_grid_params_day(
    period: Optional[int],
    seasonal: int,
    trend: Optional[int],
    low_pass: Optional[int],
    client,
):
    ts = Timeseries(
        dates=gen_dates_day(),
        name='day-test',
        data_frequency=DataFrequency.day,
    )
    ts = process_variable(ts)
    if low_pass is not None and period is not None and low_pass <= period:
        return
    params = STLParams(
        period=period,
        seasonal=seasonal,
        trend=trend,
        low_pass=low_pass,
    ).model_dump()

    data = {
        "ts": ts,
        "params": params,
    }

    result = client.post(
        url='/api/v1/generating_series/seasonal_decomposition/stl',
        json=data
    )

    data = result.json()
    if trend is not None and period is not None and trend <= period:
        assert result.status_code == 422, data
        return
    elif trend is not None and period is None and trend < 365:
        assert result.status_code == 422, data
        return
    elif low_pass is not None and period is None and low_pass < 9:
        assert result.status_code == 422, data
        return
    else:
        assert result.status_code == 200, data

    observed = data["observed"]
    trend = data["trend"]
    seasonal = data["seasonal"]
    resid = data["resid"]

    for calculated_ts in [observed, trend, seasonal, resid]:
        assert len(calculated_ts['dates']) == len(ts['dates'])
        assert delete_timestamp(calculated_ts['dates']) == ts['dates']
        assert calculated_ts['data_frequency'] == ts['data_frequency']


@pytest.mark.slow
@pytest.mark.parametrize("period", [None, 2, 7, 21, 33, 365, 377])
@pytest.mark.parametrize("seasonal", [3, 7, 21, 33, 365])
@pytest.mark.parametrize("trend", [None, 3, 13, 15, 35, 365, 377])
@pytest.mark.parametrize("low_pass", [None, 3, 13, 35, 365, 377])
def test_stl_grid_params_month(
    period: Optional[int],
    seasonal: int,
    trend: Optional[int],
    low_pass: Optional[int],
    client,
    balance
):
    ts = process_variable(balance)
    if low_pass is not None and period is not None and low_pass <= period:
        return
    params = STLParams(
        period=period,
        seasonal=seasonal,
        trend=trend,
        low_pass=low_pass,
    ).model_dump()

    data = {
        "ts": ts,
        "params": params,
    }

    result = client.post(
        url='/api/v1/generating_series/seasonal_decomposition/stl',
        json=data
    )

    data = result.json()
    if trend is not None and period is not None and trend <= period:
        assert result.status_code == 422, data
        return
    elif trend is not None and period is None and trend < 13:
        assert result.status_code == 422, data
        return
    elif low_pass is not None and period is None and low_pass < 13:
        assert result.status_code == 422, data
        return
    else:
        assert result.status_code == 200, data

    observed = data["observed"]
    trend = data["trend"]
    seasonal = data["seasonal"]
    resid = data["resid"]

    for calculated_ts in [observed, trend, seasonal, resid]:
        assert len(calculated_ts['dates']) == len(ts['dates'])
        assert delete_timestamp(calculated_ts['dates']) == ts['dates']
        assert calculated_ts['data_frequency'] == ts['data_frequency']


@pytest.mark.slow
@pytest.mark.parametrize("period", [3, 7, 21, 33])
@pytest.mark.parametrize("seasonal", [3, 7, 21, 33])
@pytest.mark.parametrize("trend", [None, 13, 15, 35])
@pytest.mark.parametrize("low_pass", [None, 13, 35])
def test_stl_grid_params_quarter(
    period: Optional[int],
    seasonal: int,
    trend: Optional[int],
    low_pass: Optional[int],
    client,
    ca
):
    pass


@pytest.mark.slow
@pytest.mark.parametrize("period", [3, 7, 21, 33])
@pytest.mark.parametrize("seasonal", [3, 7, 21, 33])
@pytest.mark.parametrize("trend", [None, 13, 15, 35])
@pytest.mark.parametrize("low_pass", [None, 13, 35])
def test_stl_grid_params_year(
    period: Optional[int],
    seasonal: int,
    trend: Optional[int],
    low_pass: Optional[int],
    client,
    u_men
):
    pass