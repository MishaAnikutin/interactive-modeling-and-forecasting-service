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
            "которое должно быть меньше `(nobs/2 - 1 - len(regression)`, "
            "где: - `nobs` — количество наблюдений в данных.  "
            "Решение: Уменьшите значение max_lags, "
            "чтобы оно соответствовало указанному ограничению, "
            "учитывая количество наблюдений и детерминированных регрессоров в вашей модели."
        )
    )

class LowCountObservationsError(BaseModel):
    type: Literal["low count observations"] = "low count observations"
    detail: str = Field(
        title="Описание ошибки",
        default="Число наблюдений слишком маленькое для выбранного regression"
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