from pydantic import BaseModel
from src.core.domain import Timeseries


class VarianceParams(BaseModel):
    timeseries: Timeseries


class VarianceResult(BaseModel):
    value: float
