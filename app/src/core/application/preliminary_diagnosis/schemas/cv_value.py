from pydantic import BaseModel
from src.core.domain import Timeseries


class VariationCoeffParams(BaseModel):
    timeseries: Timeseries


class VariationCoeffResult(BaseModel):
    value: float
