from enum import Enum
from typing import Optional

import numpy as np
from pydantic import Field, model_validator

from src.core.application.preliminary_diagnosis.schemas.common import StatTestParams, StatTestResult
from src.shared.utils import validate_float_param


class LagMethodEnum(str, Enum):
    AIC = 'AIC'
    BIC = 'BIC'
    t_stat = 't_stat'


class RegressionEnum(str, Enum):
    ConstantOnly = 'c'
    ConstantAndTrend = 'ct'
    ConstantLinearAndQuadraticTrend = 'ctt'
    NoConstantNoTrend = 'n'

class AutoLagEnum(str, Enum):
    AIC = 'AIC'
    BIC = 'BIC'
    t_stat = 't_stat'


class DickeyFullerParams(StatTestParams):
    max_lags: Optional[int] = Field(
        default=None,
        ge=0,
        title="Максимальное число, которое может быть выбрано для лага"
    )
    autolag: Optional[LagMethodEnum] = Field(
        default=LagMethodEnum.AIC,
        title="Метод выбора длины лага"
    )
    regression: RegressionEnum = Field(
        default=RegressionEnum.ConstantOnly,
        title="Компонента тренда, которую следует включить в тест"
    )

    @model_validator(mode='after')
    def validate_ts(self):
        if max(self.ts.values) == min(self.ts.values):
            raise ValueError("Invalid input, ts is constant")
        return self

    @model_validator(mode='after')
    def validate_max_lags(self):
        nobs = len(self.ts.values)
        ntrend = len(self.regression) if self.regression != "n" else 0
        if self.max_lags is None:
            # from Greene referencing Schwert 1989
            maxlag = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
            # -1 for the diff
            maxlag = min(nobs // 2 - ntrend - 1, maxlag)
            if maxlag < 0:
                raise ValueError(
                    "sample size is too short to use selected "
                    "regression component"
                )
        elif self.max_lags > nobs // 2 - ntrend - 1:
            raise ValueError(
                "maxlag must be less than (nobs/2 - 1 - ntrend) "
                "where n trend is the number of included "
                "deterministic regressors"
            )
        return self


class DickeyFullerResult(StatTestResult):
    information_criterion_max_value: Optional[float] = Field(
        title="Максимальное значение информационного критерия"
    )

    @model_validator(mode="after")
    def validate_inf_crit(self):
        self.information_criterion_max_value = validate_float_param(self.information_criterion_max_value)
        return self
