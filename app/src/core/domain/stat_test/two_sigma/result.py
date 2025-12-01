from pydantic import BaseModel, Field, model_validator

from src.shared.utils import validate_float_param
from .growth_conclusion import GrowthConclusion
from ..confidence_interval import ConfidenceInterval


class TwoSigmaTestResult(BaseModel):
    std: float = Field(..., title="Стандартное отклонение")
    confidence_interval: ConfidenceInterval = Field(..., title="Доверительный интервал")
    conclusion: GrowthConclusion = Field(..., title="Заключение")

    @model_validator(mode="after")
    def validate_value(self):
        self.std = validate_float_param(self.std)
        self.confidence_interval = tuple(validate_float_param(el) for el in self.confidence_interval)

        return self
