from pydantic import BaseModel
from src.core.domain import Timeseries


class ModeParams(BaseModel):
    timeseries: Timeseries


class ModeResult(BaseModel):
    value: float
