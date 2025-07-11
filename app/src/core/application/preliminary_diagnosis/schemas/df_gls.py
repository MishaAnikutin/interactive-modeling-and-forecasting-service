from enum import Enum
from typing import Optional

from pydantic import Field, model_validator, BaseModel

from src.core.application.preliminary_diagnosis.schemas.common import StatTestParams, CriticalValues


class TrendEnum(str, Enum):
    ConstantOnly = 'c'
    ConstantAndTrend = 'ct'

class MethodEnum(str, Enum):
    AIC = 'aic'
    BIC = 'bic'
    t_stat = 't-stat'

class DfGlsParams(StatTestParams):
    lags: Optional[int] = Field(default=None, ge=0)
    trend: TrendEnum = Field(default=TrendEnum.ConstantOnly)
    max_lags: Optional[int] = Field(default=None, ge=0)
    method: MethodEnum = Field(default=MethodEnum.AIC)

    @model_validator(mode='after')
    def validate_lags(self) -> None:
        trend_order = len(self.trend)
        lag_len = 0 if  self.lags is None else self.lags
        required = 3 + trend_order + lag_len
        if len(self.ts.values) < required:
            raise ValueError(
                f"A minimum of {required} observations are needed to run an ADF with "
                f"trend {self.trend} and the user-specified number of lags."
            )
        return self

class DfGlsResult(BaseModel):
    p_value: float
    stat_value: float
    critical_values: CriticalValues
    lags: int
    nobs: int