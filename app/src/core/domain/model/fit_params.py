from datetime import datetime

from pydantic import BaseModel, Field

from src.core.domain.timeseries.data_frequency import DataFrequency


class FitParams(BaseModel):
    val_boundary: datetime
    train_boundary: datetime
    forecast_horizon: int = Field(default=1)
    data_frequency: DataFrequency = Field(default=DataFrequency.month)

