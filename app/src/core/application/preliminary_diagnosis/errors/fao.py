from pydantic import BaseModel, Field
from typing import Union, Literal

class InvalidFreq(BaseModel):
    type: Literal["invalid freq"] = "invalid freq"
    detail: str = Field(
        default=(
            "FAO процедура определена только для месячных данных с числом точек > 24 "
            "и для дневных данных с числом точек > 400"
        ),
        title="Описание ошибки"
    )

class InvalidFreq2(BaseModel):
    type: Literal["invalid freq 2"] = "invalid freq 2"
    detail: str = Field(
        default="Заявленная частотность не соответствует выявленной в данных",
        title="Описание ошибки"
    )

class SmallSizeError(BaseModel):
    type: Literal["small size"] = "small size"
    detail: str = Field(
        default="Вычислительная ошибка, скорее всего, "
                "слишком мало наблюдений для проведения FAO процедуры",
        title="Описание ошибки"
    )

ErrorType = Union[InvalidFreq, InvalidFreq2, SmallSizeError]

class FaoValidationError(BaseModel):
    msg: ErrorType = Field(
        title="Описание ошибки",
        default=InvalidFreq()
    )