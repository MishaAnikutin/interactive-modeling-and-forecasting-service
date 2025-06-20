from pydantic import BaseModel

from src.core.domain.timeseries.timeseries import Timeseries


class Forecasts(BaseModel):
    train_predict: Timeseries
    test_predict: Timeseries
    forecast: Timeseries

