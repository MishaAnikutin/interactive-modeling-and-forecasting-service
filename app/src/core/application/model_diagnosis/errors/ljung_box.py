from typing import Annotated, Union, Literal

from pydantic import BaseModel, Field

class InvalidLagsError(BaseModel):
    type: Literal["invalid lags"] = "invalid lags"
    detail: str = Field(
        default="Значения lags должны быть больше 0 и меньше 10000",
        title="Описание ошибки"
    )


PydanticValidationErrorType = Annotated[
    Union[
        InvalidLagsError,
    ],
    Field(discriminator="type")
]


class LBPydanticValidationError(BaseModel):
    msg: PydanticValidationErrorType = Field(
        title="Описание ошибки",
        default=InvalidLagsError()
    )