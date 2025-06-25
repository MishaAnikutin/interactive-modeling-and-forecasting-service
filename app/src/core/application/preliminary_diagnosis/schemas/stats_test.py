from pydantic import BaseModel

from src.core.domain import Timeseries


class StatTestParams(BaseModel):
    alpha: float

class StatTestRequest(StatTestParams):
    ts: Timeseries

class StatTestResult(StatTestParams):
    stat_value: float
    p_value: float
