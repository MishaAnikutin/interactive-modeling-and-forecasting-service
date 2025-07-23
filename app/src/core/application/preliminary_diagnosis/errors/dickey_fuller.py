from pydantic import BaseModel, Field

from typing import Literal, Annotated, Union


class ConstantTsError(BaseModel):
    type: Literal["constant"] = "constant"
    detail: str = Field(
        title="Описание ошибки",
        default="Ряд является константой"
    )

class InvalidMaxLagsError(BaseModel):
    type: Literal["invalid max lags"] = "invalid max lags"
    detail: str = Field(
        title="Описание ошибки",
        default="maxlag must be less than (nobs/2 - 1 - ntrend) "
                "where n trend is the number of included "
                "deterministic regressors"
    )

class LowCountObservationsError(BaseModel):
    type: Literal["low count observations"] = "low count observations"
    detail: str = Field(
        title="Описание ошибки",
        default="sample size is too short to use selected "
                "regression component"
    )


PydanticValidationErrorType = Annotated[
    Union[
        InvalidMaxLagsError,
        ConstantTsError,
        LowCountObservationsError,
    ],
    Field(discriminator="type")
]


class DickeyFullerPydanticValidationError(BaseModel):
    msg: PydanticValidationErrorType = Field(
        title="Описание ошибки",
        default=LowCountObservationsError()
    )