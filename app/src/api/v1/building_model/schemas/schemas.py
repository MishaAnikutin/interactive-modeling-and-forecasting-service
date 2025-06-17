from datetime import datetime
from pydantic import BaseModel

from src.shared.schemas import Timeseries, Metric, DataFrequency


class FitParams(BaseModel):
    train_boundary: datetime
    forecast_horizon: int
    data_frequency: DataFrequency


class Forecasts(BaseModel):
    train_predict: Timeseries
    test_predict: Timeseries
    forecast: Timeseries


class ModelMetrics(BaseModel):
    train_metrics: list[Metric]
    test_metrics: list[Metric]
