from typing import Annotated, Union, Literal

from pydantic import BaseModel, Field

from src.core.application.model_diagnosis.errors.common import NotEqualFreqError, NotEqualDatesError, NotEqualLenError
from src.core.application.building_model.errors.alignment import NotLastDayOfMonthError, NotSupportedFreqError, \
    NotConstantFreqError, EmptyError, NotEqualToTargetError, NotEqualToExpectedError

class LowCountObservationsError(BaseModel):
    type: Literal["low count observations"] = "low count observations"
    detail: str = Field(
        default="Ряд должен иметь как минимум 2 наблюдения для проведения теста.",
        title="Описание ошибки"
    )


PydanticValidationErrorType = Annotated[
    Union[
        LowCountObservationsError,
    ],
    Field(discriminator="type")
]


class JarqueBeraPydanticValidationError(BaseModel):
    msg: PydanticValidationErrorType = Field(
        title="Описание ошибки",
        default=LowCountObservationsError()
    )


JBValidationErrorType = Annotated[
    Union[
        NotEqualFreqError,
        NotEqualLenError,
        NotEqualDatesError,
        NotEqualToExpectedError,
        NotEqualToTargetError,
        NotConstantFreqError,
        NotSupportedFreqError,
        NotLastDayOfMonthError,
        EmptyError,
    ],
    Field(discriminator="type")
]

class JBValidationError(BaseModel):
    msg: JBValidationErrorType = Field(
        title="Описание ошибки",
        default=NotEqualFreqError()
    )