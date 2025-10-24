from pydantic import BaseModel
from src.core.domain import Timeseries


class MeanParams(BaseModel):
    timeseries: Timeseries


class MeanResult(BaseModel):
    value: float
