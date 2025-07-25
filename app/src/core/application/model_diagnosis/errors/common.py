from typing import Literal

from pydantic import BaseModel, Field

class NotEqualFreqError(BaseModel):
    type: Literal["not equal freq"] = "not equal freq"
    detail: str = Field(
        default="Частотность данных в прогнозе не соответствует частотности в исходных",
        title="Описание ошибки"
    )

class NotEqualLenError(BaseModel):
    type: Literal["not equal lens"] = "not equal lens"
    detail: str = Field(
        default="Количество наблюдений в исходных данных и прогнозе не равны.",
        title="Описание ошибки"
    )

class NotEqualDatesError(BaseModel):
    type: Literal["not equal dates"] = "not equal dates"
    detail: str = Field(
        default="Даты в исходных данных и прогнозе не равны",
        title="Описание ошибки"
    )