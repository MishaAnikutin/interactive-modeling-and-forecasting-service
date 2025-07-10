from pydantic import BaseModel, Field

from src.core.domain import Timeseries


class CriticalValues(BaseModel):
    percent_10: float
    percent_5: float
    percent_1: float

class StatTestParams(BaseModel):
    ts: Timeseries

class StatTestResult(BaseModel):
    stat_value: float = Field(title="Значение статистики теста")
    p_value: float = Field(title="p-value")

