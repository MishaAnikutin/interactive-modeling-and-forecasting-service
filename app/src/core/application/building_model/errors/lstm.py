from typing import Annotated, Union, Literal

from pydantic import Field, BaseModel

from src.core.application.building_model.errors.alignment import NotEqualToExpectedError, \
    NotConstantFreqError, NotSupportedFreqError, NotLastDayOfMonthError, EmptyError, NoDataAfterAlignmentError, \
    BoundariesError
from src.core.application.building_model.errors.nhits import HorizonValidationError, ValSizeError, PatienceStepsError


class HiddenSizeError(BaseModel):
    type: Literal["hidden_size"] = "hidden_size"
    detail: str = Field(
        default=(
            "Размер скрытого слоя (hidden size) должен быть больше размера проекции (proj size). "
            "Размер проекции (proj size) равен 1, "
            "если параметр recurrent установлен в True, и 0 в противном случае. "
            "Решение: Убедитесь, что размер скрытого слоя больше значения proj size "
            "(1 при recurrent=True или 0 при recurrent=False)."
        ),
        title="Описание ошибки"
    )

PydanticValidationErrorType = Annotated[
    Union[
        HiddenSizeError,
        BoundariesError,
    ],
    Field(discriminator="type")
]


class LstmPydanticValidationError(BaseModel):
    msg: PydanticValidationErrorType = Field(
        title="Описание ошибки",
        default=HiddenSizeError()
    )


class LstmTrainSizeError(BaseModel):
    type: Literal["train_size"] = "train_size"
    detail: str = Field(
        default=(
            "input_size + h + размер тестовой выборки должно быть <= размер обучающей выборки"
        ),
        title="Описание ошибки"
    )

class LstmTrainSizeError2(BaseModel):
    type: Literal["train_size_2"] = "train_size_2"
    detail: str = Field(
        default=(
            "input_size + h_train + размер тестовой выборки должно быть <= размер обучающей выборки"
        )
    )

FitValidationErrorType = Annotated[
    Union[
        HorizonValidationError,
        ValSizeError,
        PatienceStepsError,
        LstmTrainSizeError,
        LstmTrainSizeError2,
        NoDataAfterAlignmentError,
        NotEqualToExpectedError,
        NotConstantFreqError,
        NotSupportedFreqError,
        NotLastDayOfMonthError,
        EmptyError
    ],
    Field(discriminator="type")
]

class LstmFitValidationError(BaseModel):
    msg: FitValidationErrorType = Field(
        title="Описание ошибки",
        default=HorizonValidationError()
    )