from typing import Annotated, Union
from pydantic import Field, BaseModel

from src.core.application.building_model.errors.alignment import NotLastDayOfMonthError, NotSupportedFreqError, \
    NotConstantFreqError, EmptyError, NotEqualToTargetError, NotEqualToExpectedError
from src.core.application.model_diagnosis.errors.common import NotEqualFreqError, NotEqualDatesError, NotEqualLenError


ArchValidationErrorType = Annotated[
    Union[
        NotEqualToExpectedError,
        NotEqualToTargetError,
        NotConstantFreqError,
        NotSupportedFreqError,
        NotLastDayOfMonthError,
        EmptyError,
        NotEqualFreqError, NotEqualLenError, NotEqualDatesError
    ],
    Field(discriminator="type")
]

class ArchValidationError(BaseModel):
    msg: ArchValidationErrorType = Field(
        title="Описание ошибки",
        default=NotEqualToExpectedError()
    )