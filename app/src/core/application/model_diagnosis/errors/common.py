from typing import Literal, Union, Annotated

from pydantic import BaseModel, Field
from src.core.application.building_model.errors.alignment import NotLastDayOfMonthError, NotSupportedFreqError, \
    NotConstantFreqError, EmptyError, NotEqualToExpectedError

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


ResidAnalysisValidationErrorType = Annotated[
    Union[
        NotEqualFreqError,
        NotEqualLenError,
        NotEqualDatesError,
        NotEqualToExpectedError,
        NotConstantFreqError,
        NotSupportedFreqError,
        NotLastDayOfMonthError,
        EmptyError,
    ],
    Field(discriminator="type")
]

class ResidAnalysisValidationError(BaseModel):
    msg: ResidAnalysisValidationErrorType = Field(
        title="Описание ошибки",
        default=NotEqualFreqError()
    )