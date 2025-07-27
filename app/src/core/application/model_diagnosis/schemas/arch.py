from typing import Optional, Literal

from pydantic import BaseModel, Field, model_validator

from src.core.application.model_diagnosis.schemas.common import ResidAnalysisData, StatTestResult
from src.shared.utils import validate_float_param


class ArchRequest(BaseModel):
    data: ResidAnalysisData = Field(
        title="Прогноз и исхдоные данные"
    )
    nlags: Optional[int] = Field(
        default=None,
        ge=0, le=10000,
        title="Максимальный лаг"
    )

    period: Optional[int] = Field(
        default=None,
        title="Период",
        ge=2, le=10000,
        description=(
            "Период сезонного временного ряда. "
            "Используется для вычисления максимального лага для сезонных "
            "данных по формуле min(2*период, nobs // 5), если указан. "
            "Если None, применяется правило по умолчанию для установки количества лагов. "
            "При задании должен быть >= 2."
        )
    )

    ddof: int = Field(
        default=0,
        title="Число степеней свободы",
        ge=0,
        le=1000
    )

    cov_type: Literal[
        "nonrobust", "fixed scale",
        "HC0", "HC1", "HC2",
        "HC3", "HAC", "hac-panel",
        "hac-groupsum", "cluster"
    ] = Field(
        default="nonrobust",
        title="Тип ковариации",
        description="Значение по умолчанию - nonrobust, который использует классическую оценку ковариации OLS. "
                    "Укажите одно из значений “HC0”, “HC1”, “HC2”, “HC3”, чтобы использовать оценку ковариации Уайта. "
                    "Принимаются все типы ковариаций, поддерживаемые OLS.fit."
    )


class ArchResult(BaseModel):
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
