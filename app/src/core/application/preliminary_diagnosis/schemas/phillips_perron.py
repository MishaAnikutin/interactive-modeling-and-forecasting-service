from enum import Enum
from typing import Optional

from pydantic import Field, model_validator

from src.core.application.preliminary_diagnosis.schemas.common import StatTestParams


class TrendEnum(str, Enum):
    ConstantOnly = 'c'
    ConstantAndTrend = 'ct'
    NoConstantNoTrend = 'n'

class TestType(str, Enum):
    tau = 'tau'
    rho = 'rho'

class PhillipsPerronParams(StatTestParams):
    lags: Optional[int] = Field(
        default=None,
        ge=0,
        title="Число лагов"
    )
    trend: TrendEnum = Field(
        default=TrendEnum.ConstantOnly,
        title="Компонента тренда, которую следует включить в тест"
    )
    test_type: TestType = Field(
        default=TestType.tau,
        title="Тип теста, который будет использоваться",
        description=('The test to use when computing the test statistic. '
                     '"tau" is based on the t-stat and "rho" uses a test based on '
                     'nobs times the re-centered regression coefficient')
    )

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
