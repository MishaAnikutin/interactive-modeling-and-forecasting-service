from typing import Optional

from pydantic import Field
from enum import Enum

from src.core.application.preliminary_diagnosis.schemas.common import StatTestParams


class RegressionEnum(str, Enum):
    ConstantOnly = "c"
    TrendOnly = "t"
    ConstantAndTrend = "ct"


class AutoLagEnum(str, Enum):
    AIC = "aic"
    BIC = "bic"
    t_stat = "t-stat"


class ZivotAndrewsParams(StatTestParams):
    lags: Optional[int] = Field(
        default=None, ge=0,
        title="Число лагов для ADF регрессии",
        description="Число лагов для ADF регрессии. Число наблюдений должно быть как минимум len(regression) + lags + 3"
    )
    max_lags: Optional[int] = Field(
        default=None,
        ge=0,
        title="Максимальное число, которое может быть выбрано для лага",
        description="Максимальное число, которое может быть выбрано для лага. max lag должен быть меньше чем число наблюдений"
    )
    trim: float = Field(
        default=0.15, ge=0.0, le=0.33,
        title="Процент серий в начале/конце, которые необходимо исключить из промежуточного периода",
        description="Процент серий в начале/конце, которые необходимо исключить из промежуточного периода"
    )
    regression: RegressionEnum = Field(
        default=RegressionEnum.ConstantOnly,
        title="Компонента тренда, которую следует включить в тест",
        description="Компонента тренда, которую следует включить в тест"
    )
    autolag: AutoLagEnum = Field(
        default=AutoLagEnum.AIC,
        title="Метод выбора длины лага",
        description="Метод выбора длины лага",
    )
