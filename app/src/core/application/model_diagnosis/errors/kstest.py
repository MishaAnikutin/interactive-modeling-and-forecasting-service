from typing import Annotated, Union, Literal

from pydantic import BaseModel, Field

class LowCountObservationsError(BaseModel):
    type: Literal["low count observations"] = "low count observations"
    detail: str = Field(
        default="Ряд должен иметь как минимум 4 наблюдения для проведения теста.",
        title="Описание ошибки"
    )


PydanticValidationErrorType = Annotated[
    Union[
        LowCountObservationsError,
    ],
    Field(discriminator="type")
]


class KolmogorovPydanticValidationError(BaseModel):
    msg: PydanticValidationErrorType = Field(
        title="Описание ошибки",
        default=LowCountObservationsError()
    )