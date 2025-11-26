from typing import Optional

from pydantic import BaseModel
from enum import Enum
from src.core.domain import Timeseries

class FaoEnum(Enum):
    normal = "Normal"
    watch_low = "Watch low"
    watch_high = "Watch high"
    alert_low = "Alert low"
    alert_high = "Alert high"


class FaoRequest(BaseModel):
    ts: Timeseries

class FaoResult(BaseModel):
    UN_result: list[Optional[FaoEnum]]
    PC_result: list[Optional[FaoEnum]]
