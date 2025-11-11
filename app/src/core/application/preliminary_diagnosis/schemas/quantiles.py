from pydantic import BaseModel
from src.core.domain import Timeseries


class QuantilesParams(BaseModel):
    timeseries: Timeseries

class QuantilesResult(BaseModel):
    q_0: float
    q_1: float
    q_5: float
    q_25: float
    q_50: float
    q_75: float
    q_95: float
    q_99: float
    q_100: float