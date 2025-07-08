from datetime import datetime

import pytest

from src.core.application.building_model.schemas.nhits import NhitsParams
from src.core.domain import FitParams, Timeseries
from tests.common.nhits import base_nhits
from tests.conftest import client

def process_fit_params(fit_params: FitParams) -> dict:
    return {
        "forecast_horizon": fit_params.forecast_horizon,
        "val_boundary": fit_params.val_boundary.strftime("%Y-%m-%d"),
        "train_boundary": fit_params.train_boundary.strftime("%Y-%m-%d"),
    }

def process_valiable(ts: Timeseries) -> dict:
    return {
        "name": ts.name,
        "values": ts.values,
        "dates": [date.strftime("%Y-%m-%d") for date in ts.dates],
        "data_frequency": ts.data_frequency,
    }


@pytest.mark.parametrize(
    "nhits_params, fit_params, dependent_variables",
    [
        (
            base_nhits,
            FitParams(),
            Timeseries()
        ),
        (
            NhitsParams(
                max_steps=500,
                early_stop_patience_steps=50,
                val_check_steps=200,
                learning_rate=5e-4,
                scaler_type="robust",
            ),
            FitParams(),
            Timeseries()
        ),
    ]
)
def test_nhits_fit_without_exog_month_frequency(
    nhits_params,
    fit_params,
    dependent_variables,
    client
):
    data = dict(
        dependent_variables=process_valiable(dependent_variables),
        explanatory_variables=None,
        hyperparameters=nhits_params.model_dump(),
        fit_params=process_fit_params(fit_params),
    )
    result = client.post(
        url='/api/v1/building_model/nhits/fit',
        json=data
    )
    assert result.status_code == 200