from enum import Enum
from typing import Optional

from pydantic import Field, model_validator

from src.core.application.preliminary_diagnosis.errors.phillips_perron import ConstantTsError, InvalidLagsError
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
        ge=0, le=10000,
        title="Число лагов",
        description="Число лагов. Число лагов должно быть меньше числа наблюдений и больше или равно 0"
    )
    trend: TrendEnum = Field(
        default=TrendEnum.ConstantOnly,
        title="Компонента тренда, которую следует включить в тест",
        description="Компонента тренда, которую следует включить в тест"
    )
    test_type: TestType = Field(
        default=TestType.tau,
        title="Тип теста, который будет использоваться",
        description=(
            "Параметр определяет тип теста для вычисления тестовой статистики: "
            "- `tau` — использует t-статистику. "
            "- `rho` — использует тест, основанный на произведении количества наблюдений (`nobs`) "
            "и рецентрированного коэффициента регрессии."
        )
    )

    @model_validator(mode='after')
    def validate_ts(self):
        if max(self.ts.values) == min(self.ts.values):
            raise ValueError(ConstantTsError().detail)
        if (self.lags is not None) and (len(self.ts.values) - 1 < self.lags):
            raise ValueError(InvalidLagsError().detail)
        return self
