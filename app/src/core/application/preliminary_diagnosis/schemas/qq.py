from pydantic import BaseModel
from src.core.domain import Timeseries


class QQParams(BaseModel):
    timeseries: Timeseries


class QQResult(BaseModel):
    data_values: list[float]
    normal_values: list[float]
