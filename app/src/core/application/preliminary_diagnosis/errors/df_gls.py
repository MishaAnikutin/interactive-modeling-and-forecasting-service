from pydantic import BaseModel, Field

from typing import Annotated, Union, Literal

class LowCountObservationsError(BaseModel):
    type: Literal["low count observations"] = "low count observations"
    detail: str = Field(
        default="Число наблюдений должно быть как минимум len(trend) + lags + 3",
        title="Описание ошибки"
    )

PydanticValidationErrorType = Annotated[
    Union[
        LowCountObservationsError,
    ],
    Field(discriminator="type")
]


class DfGlsPydanticValidationError(BaseModel):
    msg: PydanticValidationErrorType = Field(
        title="Описание ошибки",
        default=LowCountObservationsError()
    )

class SingularMatrix(BaseModel):
    type: Literal["singular matrix"] = "singular matrix"
    detail: str = Field(
        default=(
            "The maximum lag you are considering results in an ADF regression with a "
            "singular regressor matrix after including lags, and so a specification test be run. "
            "This may occur if your series have little variation and so is locally constant, "
            "or may occur if you are attempting to test a very short series. You can manually set "
            "maximum lag length to consider smaller models."
        ),
        title="Описание ошибки"
    )

class InvalidMaxLagError(BaseModel):
    type: Literal["invalid max lag"] = "invalid max lag"
    detail: str = Field(
        default="max lag должен быть меньше чем число наблюдений",
        title="Описание ошибки"
    )

ExecuteValidationErrorType = Annotated[
    Union[
        SingularMatrix, InvalidMaxLagError
    ],
    Field(discriminator="type")
]

class DfGlsExecuteValidationError(BaseModel):
    msg: ExecuteValidationErrorType = Field(
        title="Описание ошибки",
        default=SingularMatrix()
    )