from pydantic import BaseModel, Field

from typing import Annotated, Union, Literal

class InvalidLagsError(BaseModel):
    type: Literal["invalid lags"] = "invalid lags"
    detail: str = Field(
        default="Число лагов должно быть меньше числа наблюдений и больше или равно 0",
        title="Описание ошибки"
    )

class ConstantTsError(BaseModel):
    type: Literal["constant"] = "constant"
    detail: str = Field(
        title="Описание ошибки",
        default="Ряд является константой"
    )

PydanticValidationErrorType = Annotated[
    Union[
        InvalidLagsError, ConstantTsError
    ],
    Field(discriminator="type")
]


class PhillipsPerronPydanticValidationError(BaseModel):
    msg: PydanticValidationErrorType = Field(
        title="Описание ошибки",
        default=InvalidLagsError()
    )
