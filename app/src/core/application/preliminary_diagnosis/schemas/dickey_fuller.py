from enum import Enum
from typing import Optional

import numpy as np
from pydantic import Field, model_validator

from src.core.application.preliminary_diagnosis.errors.dickey_fuller import ConstantTsError, InvalidMaxLagsError, \
    LowCountObservationsError
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
        ge=0, le=10000,
        title="Максимальное число, которое может быть выбрано для лага",
        description="Максимальное число, которое может быть выбрано для лага. "
                    "Требования: max_lags < (nobs/2 - 1 - len(regression), "
                    "где: - `nobs` — количество наблюдений в данных.",
    )
    autolag: Optional[LagMethodEnum] = Field(
        default=LagMethodEnum.AIC,
        title="Метод выбора длины лага",
        description="Метод выбора длины лага"
    )
    regression: RegressionEnum = Field(
        default=RegressionEnum.ConstantOnly,
        title="Компонента тренда, которую следует включить в тест",
        description="Компонента тренда, которую следует включить в тест"
    )

    @model_validator(mode='after')
    def validate_ts(self):
        if max(self.ts.values) == min(self.ts.values):
            raise ValueError(ConstantTsError().detail)
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
                raise ValueError(LowCountObservationsError().detail)
        elif self.max_lags >= nobs // 2 - ntrend - 1:
            raise ValueError(InvalidMaxLagsError().detail)
        return self


class DickeyFullerResult(StatTestResult):
    information_criterion_max_value: Optional[float] = Field(
        title="Максимальное значение информационного критерия",
        description="Максимальное значение информационного критерия",
    )

    @model_validator(mode="after")
    def validate_inf_crit(self):
        self.information_criterion_max_value = validate_float_param(self.information_criterion_max_value)
        return self
