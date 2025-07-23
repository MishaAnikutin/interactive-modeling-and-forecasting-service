from pydantic import BaseModel, Field

from typing import Annotated, Union, Literal

class LowCountObservationsError(BaseModel):
    type: Literal["low count observations"] = "low count observations"
    detail: str = Field(
        default="Число наблюдений должно быть как минимум 25",
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
        LowCountObservationsError, ConstantTsError
    ],
    Field(discriminator="type")
]


class RangePydanticValidationError(BaseModel):
    msg: PydanticValidationErrorType = Field(
        title="Описание ошибки",
        default=LowCountObservationsError()
    )
