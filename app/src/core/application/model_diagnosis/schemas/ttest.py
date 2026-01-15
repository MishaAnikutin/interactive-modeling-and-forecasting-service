from pydantic import BaseModel

from src.core.domain import Timeseries
from src.core.domain.stat_test import SignificanceLevel
from src.core.domain.stat_test.nan_policy import NanPolicy
from src.core.domain.stat_test.alternative import Alternative


class TtestRequest(BaseModel):
    forecast: Timeseries
    actual: Timeseries
    significance_level: SignificanceLevel = 0.05
    alternative: Alternative = Alternative.TWO_SIDED
    nan_policy: NanPolicy = NanPolicy.PROPAGATE
