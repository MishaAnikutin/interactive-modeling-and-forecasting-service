from pydantic import BaseModel, Field

from typing import Annotated, Union, Literal

class SingularMatrix(BaseModel):
    type: Literal["singular matrix"] = "singular matrix"
    detail: str = Field(
        default=(
            f"The regressor matrix is singular. The can happen if the data "
            "contains regions of constant observations, if the number of "
            f"lags is too large, or if the series is very "
            "short."
        ),
        title="Описание ошибки"
    )

class SingularMatrix2(BaseModel):
    type: Literal["singular matrix 2"] = "singular matrix 2"
    detail: str = Field(
        default=(
            "The maximum lag you are considering results in an ADF regression with a"
            "singular regressor matrix after including lags, and so a specification test be "
            "run. This may occur if your series have little variation and so is locally constant,"
            "or may occur if you are attempting to test a very short series. You can manually set"
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

class LowCountObservationsError(BaseModel):
    type: Literal["low count observations"] = "low count observations"
    detail: str = Field(
        default="Слишком мало наблюдений для реализации теста с данными trend и lags"
    )

class LowCountObservationsError2(BaseModel):
    type: Literal["low count observations 2"] = "low count observations 2"
    detail: str = Field(
        default="Число наблюдений должно быть как минимум len(trend) + lags + 3",
        title="Описание ошибки"
    )

ExecuteValidationErrorType = Annotated[
    Union[
        SingularMatrix,
        SingularMatrix2,
        InvalidMaxLagError,
        LowCountObservationsError,
        LowCountObservationsError2
    ],
    Field(discriminator="type")
]

class ZivotAndrewsExecuteValidationError(BaseModel):
    msg: ExecuteValidationErrorType = Field(
        title="Описание ошибки",
        default=SingularMatrix()
    )