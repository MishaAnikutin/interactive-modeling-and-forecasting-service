from pydantic import BaseModel
from src.core.domain import Timeseries


class KurtosisParams(BaseModel):
    timeseries: Timeseries


class KurtosisResult(BaseModel):
    value: float
