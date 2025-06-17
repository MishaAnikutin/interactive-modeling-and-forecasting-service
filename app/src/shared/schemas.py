from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class Timeseries(BaseModel):
    dates: list[datetime]
    values: list[Optional[float]]


class Metric(BaseModel):
    type: str
    value: float


class Coefficient(BaseModel):
    name: str
    value: float
    p_value: float


class DataFrequency(str, Enum):
    year: str = "Y"
    month: str = "M"
    quart: str = "Q"
    day: str = "D"
    hour: str = "H"
    minute: str = "M"
