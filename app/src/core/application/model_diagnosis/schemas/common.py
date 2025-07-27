from typing import Optional, List

from pydantic import BaseModel, Field, model_validator

from src.core.domain import Forecasts, Timeseries
from src.shared.utils import validate_float_param


class ResidAnalysisData(BaseModel):
    """Прогнозы и исходные данные"""
    forecasts: Forecasts = Field(
        title="Прогнозы",
        description="Даты объединенного прогноза должны совпадать с датами исторических данных."
    )

    target: Timeseries = Field(
        title="Исходные данные",
        default=Timeseries(name="Исходные данные")
    )

    exog: Optional[List[Timeseries]] = Field(
        default=None,
        title="Экзогенные переменные (опционально)"
    )


class StatTestResult(BaseModel):
    """Результаты статистического теста"""
    p_value: Optional[float] = Field(
        default=0.05,
        title="p-value теста",
        description="p-value теста"
    )
    stat_value: Optional[float] = Field(
        title="Значение статистики теста",
        description="Значение статистики теста"
    )

    @model_validator(mode='after')
    def validate_float(self):
        self.p_value = validate_float_param(self.p_value)
        self.stat_value = validate_float_param(self.stat_value)
        return self