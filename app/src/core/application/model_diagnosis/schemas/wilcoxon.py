from typing import Optional

from pydantic import BaseModel

from src.core.domain import Timeseries, Forecasts
from src.core.domain.stat_test import SignificanceLevel
from src.core.domain.stat_test.alternative import Alternative
from src.core.domain.stat_test.method import Method
from src.core.domain.stat_test.nan_policy import NanPolicy
from src.core.domain.stat_test.wilcoxon.params import ZeroMethod
from src.core.domain.stat_test.wilcoxon.result import WilcoxonResult


class WilcoxonRequest(BaseModel):
    forecasts: Forecasts
    target: Timeseries
    significance_level: SignificanceLevel = 0.05
    correction: bool = False
    alternative: Alternative = Alternative.TWO_SIDED
    nan_policy: NanPolicy = NanPolicy.PROPAGATE
    method: Method = Method.AUTO
    zero_method: ZeroMethod = ZeroMethod.wilcox


class WilcoxonResponse(BaseModel):
    train_result: WilcoxonResult
    validation_result: Optional[WilcoxonResult]
    test_result: Optional[WilcoxonResult]
