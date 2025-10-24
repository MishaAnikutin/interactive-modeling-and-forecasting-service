from pydantic import BaseModel
from src.core.domain import Timeseries


class MedianParams(BaseModel):
    timeseries: Timeseries


class MedianResult(BaseModel):
    value: float
