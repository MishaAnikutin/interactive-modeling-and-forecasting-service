from typing import Optional

from pydantic import BaseModel

from src.core.domain import Timeseries, Forecasts
from src.core.domain.stat_test import SignificanceLevel
from src.core.domain.stat_test.mann_whitney.result import MannWhitneyResult
from src.core.domain.stat_test.nan_policy import NanPolicy
from src.core.domain.stat_test.alternative import Alternative
from src.core.domain.stat_test.method import Method


class MannWhitneyRequest(BaseModel):
    forecasts: Forecasts
    target: Timeseries
    significance_level: SignificanceLevel = 0.05
    use_continuity: bool = True
    alternative: Alternative = Alternative.TWO_SIDED
    method: Method = Method.AUTO
    nan_policy: NanPolicy = NanPolicy.PROPAGATE


class MannWhitneyResponse(BaseModel):
    train_result: MannWhitneyResult
    validation_result: Optional[MannWhitneyResult]
    test_result: Optional[MannWhitneyResult]

