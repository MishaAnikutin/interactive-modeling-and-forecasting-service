from typing import Annotated, Union, Literal

from pydantic import BaseModel, Field


class NotOddError(BaseModel):
    type: Literal["not odd"] = "not odd"
    detail: str = Field(
        default="Параметры trend, seasonal и low_pass должны быть нечетными",
        title="Описание ошибки"
    )

class InvalidLowPassError(BaseModel):
    type: Literal["invalid low pass"] = "invalid low pass"
    detail: str = Field(
        default=(
            "Low pass должен быть целым число >= 3, которое больше period, если period указан."
            "Если period не указан, то: "
            "1) при дневной частотности low_pass >= 9 "
            "2) при месячной частотности low_pass >= 13 "
            "3) при квартальной частотности low_pass >= 5 "
            "4) при годовой частотности надо выбрать period согласно правилу выше"
        ),
        title="Описание ошибки"
    )

class InvalidTrendError(BaseModel):
    type: Literal["invalid trend"] = "invalid trend"
    detail: str = Field(
        default=(
            "Trend должен быть целым число >= 3, которое больше period, если period указан."
            "Если period не указан, то: "
            "1) при дневной частотности period >= 365 "
            "2) при месячной частотности period >= 13 "
            "3) при квартальной частотности period >= 5 "
            "4) при годовой частотности надо выбрать period согласно правилу выше"
        ),
        title="Описание ошибки"
    )

class NonePeriodError(BaseModel):
    type: Literal["none period"] = "none period"
    detail: str = Field(
        default="Если ряд имеет годовую частотность, то period должен быть не пустым",
        title="Описание ошибки"
    )


PydanticValidationErrorType = Annotated[
    Union[
        NotOddError,
        InvalidLowPassError,
        NonePeriodError
    ],
    Field(discriminator="type")
]


class STLDecompPydanticValidationError(BaseModel):
    msg: PydanticValidationErrorType = Field(
        title="Описание ошибки",
        default=InvalidLowPassError()
    )