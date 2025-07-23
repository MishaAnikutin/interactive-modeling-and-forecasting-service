from typing import Annotated, Union

from pydantic import Field, BaseModel

from src.core.application.building_model.errors.alignment import NotLastDayOfMonthError, NotSupportedFreqError, \
    NotConstantFreqError, EmptyError, NotEqualToTargetError, NotEqualToExpectedError

FitValidationErrorType = Annotated[
    Union[
        NotEqualToExpectedError,
        NotEqualToTargetError,
        NotConstantFreqError,
        NotSupportedFreqError,
        NotLastDayOfMonthError,
        EmptyError
    ],
    Field(discriminator="type")
]

class ArimaxFitValidationError(BaseModel):
    msg: FitValidationErrorType = Field(
        title="Описание ошибки",
        default=NotEqualToExpectedError()
    )