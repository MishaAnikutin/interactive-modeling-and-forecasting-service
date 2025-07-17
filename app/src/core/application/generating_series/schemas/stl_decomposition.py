from typing import Optional

from pydantic import BaseModel, Field

from src.core.domain import Timeseries


class STLParams(BaseModel):
    period: Optional[int] = Field(default=None)
    seasonal: int = Field(default=7)
    trend: Optional[int] = Field(default=None)
    low_pass: Optional[int] = Field(default=None)
    seasonal_deg: int = Field(default=1)
    trend_deg: int = Field(default=1)
    low_pass_deg: int = Field(default=1)
    robust: bool = Field(default=False)
    seasonal_jump: int = Field(default=1)
    trend_jump: int = Field(default=1)
    low_pass_jump: int = Field(default=1)

class STLDecompositionRequest(BaseModel):
    ts: Timeseries
    params: STLParams

class STLDecompositionResult(BaseModel):
    observed: Optional[Timeseries]
    seasonal: Optional[Timeseries]
    trend: Optional[Timeseries]
    resid: Optional[Timeseries]
