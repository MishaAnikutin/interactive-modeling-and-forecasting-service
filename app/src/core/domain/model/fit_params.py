from datetime import datetime

from pydantic import BaseModel

from src.core.domain.timeseries.data_frequency import DataFrequency


class FitParams(BaseModel):
    train_boundary: datetime
    forecast_horizon: int
    data_frequency: DataFrequency

