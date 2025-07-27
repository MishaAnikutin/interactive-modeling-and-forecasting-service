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

class DiagnosticsResult(BaseModel):
    lmval: Optional[float] = Field(
        title="LM-статистика",
        description="Тестовая статистика множителя Лагранжа, используемая для проверки ограничений в модели.",
    )
    lmpval: Optional[float] = Field(
        title="P-значение LM",
        description="P-значение теста множителя Лагранжа, показывающее вероятность ошибки при отклонении нулевой гипотезы.",
    )
    fval: Optional[float] = Field(
        title="F-статистика",
        description="F-статистика теста, альтернативная версия теста множителя Лагранжа, основанная на F-тесте для проверки ограничений параметров.",
    )
    fpval: Optional[float] = Field(
        title="P-значение F",
        description="P-значение F-теста, указывающее на значимость ограничений в модели.",
    )

    @model_validator(mode="after")
    def validate_floats(self):
        self.lmval = validate_float_param(self.lmval)
        self.lmpval = validate_float_param(self.lmpval)
        self.fval = validate_float_param(self.fval)
        self.fpval = validate_float_param(self.fpval)
        return self