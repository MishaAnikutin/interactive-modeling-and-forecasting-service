from typing import Annotated, Union, Literal

from pydantic import Field, BaseModel

from src.core.application.building_model.errors.alignment import (
    EmptyError,
    NotConstantFreqError,
    NotEqualToTargetError,
    NotSupportedFreqError,
    NotLastDayOfMonthError,
    NotEqualToExpectedError,
)


class ConstantInExogAndSpecificationError(BaseModel):
    type: Literal["constant in exog and specification"] = "constant in exog and specification"
    detail: str = Field(
        default="Модель включает в себя константу тренда, однако экзогенные переменные также включают константный ряд",
        title="Описание ошибки"
    )


FitValidationErrorType = Annotated[
    Union[
        NotEqualToExpectedError,
        NotEqualToTargetError,
        NotConstantFreqError,
        NotSupportedFreqError,
        NotLastDayOfMonthError,
        EmptyError,
        ConstantInExogAndSpecificationError
    ],
    Field(discriminator="type")
]


class ArimaxFitValidationError(BaseModel):
    msg: FitValidationErrorType = Field(
        title="Описание ошибки",
        default=NotEqualToExpectedError()
    )
