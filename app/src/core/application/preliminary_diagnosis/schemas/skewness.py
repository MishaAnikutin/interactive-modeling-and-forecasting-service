from pydantic import BaseModel
from src.core.domain import Timeseries


class SkewnessParams(BaseModel):
    timeseries: Timeseries


class SkewnessResult(BaseModel):
    value: float
