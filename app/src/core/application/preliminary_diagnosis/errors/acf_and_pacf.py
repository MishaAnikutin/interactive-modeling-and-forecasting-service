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
        default=(
            "Ошибка указывает, что максимальное количество лагов max_lags превышает допустимое значение, "
            "которое должно быть меньше `(nobs/2 - 1)`, "
            "где: - `nobs` — количество наблюдений в данных.  "
            "Решение: Уменьшите значение max_lags, "
            "чтобы оно соответствовало указанному ограничению."
        )
    )


class LowCountObservationsError(BaseModel):
    type: Literal["low count observations"] = "low count observations"
    detail: str = Field(
        title="Описание ошибки",
        default="Число наблюдений слишком маленькое для выбранного количества лагов"
    )


class EmptyTimeSeriesError(BaseModel):
    type: Literal["empty time series"] = "empty time series"
    detail: str = Field(
        title="Описание ошибки",
        default="Временной ряд не может быть пустым"
    )


class InvalidAlphaError(BaseModel):
    type: Literal["invalid alpha"] = "invalid alpha"
    detail: str = Field(
        title="Описание ошибки",
        default="Уровень значимости alpha должен быть между 0 и 1"
    )


class NaNValuesError(BaseModel):
    type: Literal["nan values"] = "nan values"
    detail: str = Field(
        title="Описание ошибки",
        default="Временной ряд содержит пропущенные значения (NaN)"
    )


class NonNumericDataError(BaseModel):
    type: Literal["non numeric data"] = "non numeric data"
    detail: str = Field(
        title="Описание ошибки",
        default="Временной ряд должен содержать только числовые значения"
    )


AcfPacfValidationErrorType = Annotated[
    Union[
        InvalidMaxLagsError,
        ConstantTsError,
        LowCountObservationsError,
        EmptyTimeSeriesError,
        InvalidAlphaError,
        NaNValuesError,
        NonNumericDataError,
    ],
    Field(discriminator="type")
]


class AcfPacfValidationError(BaseModel):
    error: AcfPacfValidationErrorType = Field(
        title="Ошибка валидации ACF/PACF",
        description="Детализированная информация об ошибке"
    )