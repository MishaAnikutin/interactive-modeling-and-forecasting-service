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
            "Ошибка возникает, когда в регрессии ADF (тест Дики-Фуллера) матрица регрессоров становится сингулярной из-за выбранного максимального лага. Это может произойти, если:"
            "1. Ваши данные имеют низкую вариацию и выглядят почти постоянными."
            "2. Вы анализируете слишком короткий временной ряд."
            "Решение: Уменьшите максимальное количество лагов вручную, чтобы использовать меньшие модели."
        ),
        title="Описание ошибки"
    )

class InvalidMaxLagError(BaseModel):
    type: Literal["invalid max lag"] = "invalid max lag"
    detail: str = Field(
        default="max lag должен быть меньше, чем число наблюдений",
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