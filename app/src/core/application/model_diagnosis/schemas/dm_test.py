from pydantic import BaseModel
from src.core.domain import Timeseries
from src.core.domain.stat_test import SignificanceLevel
from src.core.domain.stat_test.dm_test.variance_estimator import VarianceEstimator
from src.core.domain.stat_test.nan_policy import NanPolicy
from src.core.domain.stat_test.alternative import Alternative
from src.core.domain.stat_test.dm_test.loss_function import LossFunction


class DmTestRequest(BaseModel):
    forecast1: Timeseries
    forecast2: Timeseries
    actual: Timeseries
    significance_level: SignificanceLevel = 0.05
    h: int = 1
    one_sided: bool = False
    harvey_correction: bool = True
    variance_estimator: VarianceEstimator
