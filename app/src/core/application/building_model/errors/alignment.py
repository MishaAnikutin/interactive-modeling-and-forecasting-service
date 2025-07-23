from typing import Literal

from pydantic import BaseModel, Field


class NotEqualToExpectedError(BaseModel):
    type: Literal["equal"] = "equal"
    detail: str = Field(
        default="Не соответствует полученный тип частотности ряда и заявленный для переменной",
        title="Описание ошибки"
    )

class NotEqualToTargetError(BaseModel):
    type: Literal["target"] = "target"
    detail: str = Field(
        default="Частотность экзогенной переменной не соответствует частотности целевой переменной.",
        title="Описание ошибки"
    )

class NotConstantFreqError(BaseModel):
    type: Literal["not constant frequency"] = "not constant frequency"
    detail: str = Field(
        default="Ряд не постоянной частотности",
        title="Описание ошибки"
    )

class NotSupportedFreqError(BaseModel):
    type: Literal["not supported frequency"] = "not supported frequency"
    detail: str = Field(
        default="Ряд имеет неподдерживаемую частотность. Разрешенные: [Y, Q, M, D]",
        title="Описание ошибки"
    )

class NotLastDayOfMonthError(BaseModel):
    type: Literal["not last day of month"] = "not last day of month"
    detail: str = Field(
        default="Дата не является последним днем месяца",
        title="Описание ошибки"
    )

class EmptyError(BaseModel):
    type: Literal["empty"] = "empty"
    detail: str = Field(
        default="Ряд должен быть не пустой",
        title="Описание ошибки"
    )