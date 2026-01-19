from typing import Optional

from pydantic import BaseModel
from src.core.domain import Timeseries, Forecasts
from src.core.domain.stat_test import SignificanceLevel
from src.core.domain.stat_test.dm_test.result import DmTestResult
from src.core.domain.stat_test.dm_test.variance_estimator import VarianceEstimator
from src.core.domain.stat_test.nan_policy import NanPolicy
from src.core.domain.stat_test.alternative import Alternative
from src.core.domain.stat_test.dm_test.loss_function import LossFunction


class DmTestRequest(BaseModel):
    forecasts1: Forecasts
    forecasts2: Forecasts
    target: Timeseries
    significance_level: SignificanceLevel = 0.05
    h: int = 1
    one_sided: bool = False
    harvey_correction: bool = True
    variance_estimator: VarianceEstimator


class DmTestResponse(BaseModel):
    train_result: DmTestResult
    validation_result: Optional[DmTestResult]
    test_result: Optional[DmTestResult]
