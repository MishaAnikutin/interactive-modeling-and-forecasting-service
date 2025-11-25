from typing import Optional

from pydantic import BaseModel, Field, model_validator

from src.core.domain import Timeseries
from src.shared.utils import validate_float_param


class CriticalValues(BaseModel):
    percent_10: Optional[float] = Field(title="Критическое значение статистики для 10%")
    percent_5: Optional[float] = Field(title="Критическое значение статистики для 5%")
    percent_1: Optional[float] = Field(title="Критическое значение статистики для 1%")

    @model_validator(mode='after')
    def validate_float(self):
        self.percent_10 = validate_float_param(self.percent_10)
        self.percent_5 = validate_float_param(self.percent_5)
        self.percent_1 = validate_float_param(self.percent_1)
        return self

class StatTestParams(BaseModel):
    ts: Timeseries = Field(
        default=Timeseries(name="Временной ряд для анализа"),
        title="Временной ряд для анализа",
        description="Временной ряд для анализа",
    )


class ResultValues(BaseModel):
    p_value: Optional[float] = Field(
        default=0.05,
        title="p-value теста",
        description="p-value теста"
    )
    stat_value: Optional[float] = Field(
        title="Значение статистики теста",
        description="Значение статистики теста"
    )


class StatTestResult(ResultValues):
    critical_values: CriticalValues = Field(
        title="Критические значения для разных уровней значимости",
        description="Критические значения для разных уровней значимости",
    )
    lags: int = Field(
        title="Число лагов, использованных при расчетах",
        description="Число лагов, использованных при расчетах",
    )

    @model_validator(mode='after')
    def validate_float(self):
        self.p_value = validate_float_param(self.p_value)
        self.stat_value = validate_float_param(self.stat_value)
        return self

