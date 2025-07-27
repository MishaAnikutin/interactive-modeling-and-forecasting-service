from typing import Optional, List

from pydantic import BaseModel, Field, model_validator

from src.core.application.model_diagnosis.schemas.common import ResidAnalysisData
from src.shared.utils import validate_float_param


class BreuschGodfreyRequest(BaseModel):
    data: ResidAnalysisData = Field(
        title="Прогнозы и исходные данные"
    )
    nlags: Optional[int] = Field(
        default=None,
        ge=0, le=10000,
        title="Число лагов"
    )


class BreuschGodfreyResult(BaseModel):
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