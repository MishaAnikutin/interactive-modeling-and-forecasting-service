from pydantic import BaseModel, Field

from typing import Annotated, Union, Literal

class InvalidLagsError(BaseModel):
    type: Literal["invalid lags"] = "invalid lags"
    detail: str = Field(
        default="Число лагов должно быть меньше числа наблюдений и больше или равно 0",
        title="Описание ошибки"
    )

PydanticValidationErrorType = Annotated[
    Union[
        InvalidLagsError,
    ],
    Field(discriminator="type")
]


class KpssPydanticValidationError(BaseModel):
    msg: PydanticValidationErrorType = Field(
        title="Описание ошибки",
        default=InvalidLagsError()
    )
