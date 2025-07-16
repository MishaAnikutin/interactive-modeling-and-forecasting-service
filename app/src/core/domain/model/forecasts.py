from typing import Optional

from pydantic import BaseModel

from src.core.domain.timeseries.timeseries import Timeseries


class Forecasts(BaseModel):
    train_predict: Timeseries
    validation_predict: Optional[Timeseries]
    test_predict: Optional[Timeseries]
    forecast: Optional[Timeseries]

