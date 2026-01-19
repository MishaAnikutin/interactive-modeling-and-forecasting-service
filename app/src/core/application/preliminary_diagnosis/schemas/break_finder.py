from datetime import datetime
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from src.core.domain import Timeseries
from src.core.domain.structural_shift.break_criterion import BreakCriterion


class BreakFinderRequest(BaseModel):
    timeseries: Timeseries
    trim: tuple[Union[float, int], Union[float, int]] = (0.15, 0.15)
    gap: float = 0.15

    n_breaks: int = Field(default=1, description="число сдвигов")
    criterion: BreakCriterion
    intercept: bool = True
    break_intercept: bool = True
    trend: bool = False
    break_trend: bool = False
    seasons: int = 0


class BreakFinderResponse(BaseModel):
    break_datetimes: List[datetime]
