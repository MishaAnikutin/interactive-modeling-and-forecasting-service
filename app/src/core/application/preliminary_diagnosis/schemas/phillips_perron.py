from enum import Enum
from typing import Optional

from pydantic import Field, model_validator, BaseModel

from src.core.application.preliminary_diagnosis.schemas.common import StatTestParams, CriticalValues


class TrendEnum(str, Enum):
    ConstantOnly = 'c'
    ConstantAndTrend = 'ct'
    NoConstantNoTrend = 'n'

class TestType(str, Enum):
    tau = 'tau'
    rho = 'rho'

class PhillipsPerronParams(StatTestParams):
    lags: Optional[int] = Field(default=None, ge=0)
    trend: TrendEnum = Field(default=TrendEnum.ConstantOnly)
    test_type: TestType = Field(default=TestType.tau)

    @model_validator(mode='after')
    def validate_ts(self):
        if max(self.ts.values) == min(self.ts.values):
            raise ValueError("Invalid input, ts is constant")
        if (self.lags is not None) and (len(self.ts.values) - 1 < self.lags):
            raise ValueError(
                f"The number of observations {len(self.ts.values)} is less than the number of"
                f"lags in the long-run covariance estimator, {self.lags}. You must have "
                "lags <= nobs."
            )
        return self

class PhillipsPerronResult(BaseModel):
    p_value: float
    stat_value: float
    critical_values: CriticalValues
    lags: int
    nobs: int