from enum import Enum
from typing import Optional

from pydantic import Field, model_validator

from src.core.application.preliminary_diagnosis.errors.df_gls import LowCountObservationsError
from src.core.application.preliminary_diagnosis.schemas.common import StatTestParams


class TrendEnum(str, Enum):
    ConstantOnly = 'c'
    ConstantAndTrend = 'ct'

class MethodEnum(str, Enum):
    AIC = 'aic'
    BIC = 'bic'
    t_stat = 't-stat'

class DfGlsParams(StatTestParams):
    lags: Optional[int] = Field(
        default=None, ge=0, le=10000,
        title="Число лагов для ADF регрессии",
        description=(
            "Число лагов для ADF регрессии. "
            "Требования:"
            "1) Число наблюдений должно быть как минимум len(trend) + lags + 3"
        )
    )
    trend: TrendEnum = Field(
        default=TrendEnum.ConstantOnly,
        title="Компонента тренда, которую следует включить в тест",
        description=(
            "Компонента тренда, которую следует включить в тест "
            "Требования:"
            "1) Число наблюдений должно быть как минимум len(trend) + lags + 3"
        ),
    )
    max_lags: Optional[int] = Field(
        default=None,
        ge=0, le=10000,
        title="Максимальное число, которое может быть выбрано для лага",
        description=(
            "Максимальное число, которое может быть выбрано для лага. "
            "max lag должен быть меньше, чем число наблюдений"
        )
    )
    method: MethodEnum = Field(
        default=MethodEnum.AIC,
        title="Метод выбора длины лага",
        description="Метод выбора длины лага",
    )

    @model_validator(mode='after')
    def validate_lags(self) -> None:
        trend_order = len(self.trend)
        lag_len = 0 if  self.lags is None else self.lags
        required = 3 + trend_order + lag_len
        if len(self.ts.values) < required:
            raise ValueError(LowCountObservationsError().detail)
        return self
