from typing import Annotated, Union, Literal

from pydantic import BaseModel, Field


class MissingError(BaseModel):
    type: Literal["missing values"] = "missing values"
    detail: str = Field(
        default="Эта функция не обрабатывает ряды с пустыми значениями",
        title="Описание ошибки"
    )


class ZeroOrNegativeError(BaseModel):
    type: Literal["zero or negative"] = "zero or negative"
    detail: str = Field(
        default="Эта функция не обрабатывает ряды с нулевыми или отрицательными значениями",
        title="Описание ошибки"
    )


class LowCountObservationsError(BaseModel):
    type: Literal["low count observations"] = "low count observations"
    detail: str = Field(
        default="Ряд должен иметь как минимум 2 * period наблюдения, "
                "где period - это сколько раз в году проходит сезон ряда "
                "(например для квартальных данных это 4)",
        title="Описание ошибки"
    )

class LowCountObservationsError2(BaseModel):
    type: Literal["low count observations 2"] = "low count observations 2"
    detail: str = Field(
        default="Длина списка filt (len(filt)) должна быть меньше или равна числа наблюдений",
        title="Описание ошибки"
    )


PydanticValidationErrorType = Annotated[
    Union[
        MissingError,
        ZeroOrNegativeError,
        LowCountObservationsError,
        LowCountObservationsError2
    ],
    Field(discriminator="type")
]


class NaiveDecompPydanticValidationError(BaseModel):
    msg: PydanticValidationErrorType = Field(
        title="Описание ошибки",
        default=LowCountObservationsError()
    )