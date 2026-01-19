from typing import Optional
from pydantic import BaseModel

from src.core.domain import Timeseries, Forecasts
from src.core.domain.stat_test import SignificanceLevel
from src.core.domain.stat_test.nan_policy import NanPolicy
from src.core.domain.stat_test.alternative import Alternative
from src.core.domain.stat_test.ttest.result import TtestResult


class TtestRequest(BaseModel):
    forecasts: Forecasts
    target: Timeseries
    significance_level: SignificanceLevel = 0.05
    alternative: Alternative = Alternative.TWO_SIDED
    nan_policy: NanPolicy = NanPolicy.PROPAGATE


class TtestResponse(BaseModel):
    train_result: TtestResult
    validation_result: Optional[TtestResult]
    test_result: Optional[TtestResult]
