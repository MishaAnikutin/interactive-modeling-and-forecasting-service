from pydantic import BaseModel, Field

from typing import Annotated, Union, Literal

class SingularMatrix(BaseModel):
    type: Literal["singular matrix"] = "singular matrix"
    detail: str = Field(
        default=(
            "Матрица регрессоров является сингулярной. Это может произойти, если: "
            "1. Данные содержат участки с постоянными значениями. "
            "2. Количество лагов слишком велико. "
            "3. Временной ряд слишком короткий. "
            "Решение: Проверьте данные на наличие постоянных участков, "
            "уменьшите количество лагов или увеличьте длину временного ряда."
        ),
        title="Описание ошибки"
    )

class SingularMatrix2(BaseModel):
    type: Literal["singular matrix 2"] = "singular matrix 2"
    detail: str = Field(
        default=(
            "Ошибка возникает, когда в регрессии ADF (тест Дики-Фуллера) матрица регрессоров становится сингулярной из-за выбранного максимального лага. Это может произойти, если: "
            "1. Ваши данные имеют низкую вариацию и выглядят почти постоянными. "
            "2. Вы анализируете слишком короткий временной ряд. "
            "Решение: Уменьшите максимальное количество лагов вручную, чтобы использовать меньшие модели."
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
        default="Слишком мало наблюдений для реализации теста с данными regression и lags"
    )

class LowCountObservationsError2(BaseModel):
    type: Literal["low count observations 2"] = "low count observations 2"
    detail: str = Field(
        default="Число наблюдений должно быть как минимум len(regression) + lags + 3",
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