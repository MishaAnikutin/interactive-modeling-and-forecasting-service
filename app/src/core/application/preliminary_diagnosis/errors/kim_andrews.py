from pydantic import BaseModel, Field
from typing import Union, Literal

class LargeShiftError(BaseModel):
    type: Literal["large shift"] = "large shift"
    detail: str = Field(
        default="shift должен быть меньше длины ряда",
        title="Описание ошибки"
    )

class NotEnoughPointsError(BaseModel):
    type: Literal["not enough points"] = "not enough points"
    detail: str = Field(
        default="После сдвига остаётся только {число} наблюдений, а требуется минимум n + m",
        title="Описание ошибки"
    )

class SmallSizeError(BaseModel):
    type: Literal["small n"] = "small n"
    detail: str = Field(
        default="Необходимо n > m + 1",
        title="Описание ошибки"
    )

ErrorType = Union[LargeShiftError, NotEnoughPointsError, SmallSizeError]

class KimAndrewsValidationError(BaseModel):
    msg: ErrorType = Field(
        title="Описание ошибки",
        default=NotEnoughPointsError()
    )