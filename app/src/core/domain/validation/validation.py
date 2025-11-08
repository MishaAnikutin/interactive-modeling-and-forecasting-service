from enum import StrEnum

from pydantic import BaseModel, Field


class ValidationType(StrEnum):
    """Стратегии валидации рядов"""

    EXCESSIVE_ZEROS = "EXCESSIVE_ZEROS"
    SPARSE_DATA = "SPARSE_DATA"
    SINGLE_OBSERVATION_FREQUENCY = "SINGLE_OBSERVATION_FREQUENCY"
    UNDETERMINED_FREQUENCY = "UNDETERMINED_FREQUENCY"
    FUTURE_VALUES = "FUTURE_VALUES"
    MIXED_FREQUENCY = "MIXED_FREQUENCY"


class ValidationIssue(BaseModel):
    type: ValidationType
    is_severity: bool = Field(..., title="Серьезность наличия ошибки валидации.",
                              description="Если True, то ряд нельзя использовать в прогнозных моделях")
    message: str = Field(..., title="Описание ошибки валидации")
