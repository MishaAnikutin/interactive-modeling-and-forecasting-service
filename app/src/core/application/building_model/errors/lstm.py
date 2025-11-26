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
            f"Недостаточный размер обучающей выборки для LSTM / GRU модели. "
            f"Получено: train_size=, input_size=, h=, test_size=. "
            f"Требуется: train_size >= (input_size + h + test_size)"
        ),
        title="Описание ошибки обучающей выборки"
    )

    def __init__(self, input_size: int, h: int, test_size: int, train_size: int, **data):
        required_min = input_size + h

        detail_msg = (
            f"Недостаточный размер обучающей выборки для LSTM / GRU модели. "
            f"Получено: train_size={train_size}, input_size={input_size}, h={h} (горизонт прогноза), test_size={test_size}. "
            f"Требуется: train_size >= {required_min} (input_size + h + test_size) "
        )

        super().__init__(detail=detail_msg, **data)

class LstmTrainSizeError2(BaseModel):
    type: Literal["train_size_2"] = "train_size_2"
    detail: str = Field(
        default=(
            f"Недостаточный размер обучающей выборки для LSTM / GRU модели. "
            f"Получено: train_size=, input_size=, h_train=, test_size=. "
            f"Требуется: train_size >= (input_size + h_train + test_size)"
        ),
        title="Описание ошибки обучающей выборки"
    )

    def __init__(self, input_size: int, h_train: int, test_size: int, train_size: int, **data):
        required_min = input_size + h_train + test_size

        detail_msg = (
            f"Недостаточный размер обучающей выборки для LSTM / GRU модели. "
            f"Получено: train_size={train_size}, input_size={input_size}, h_train={h_train}, test_size={test_size}. "
            f"Требуется: train_size >= {required_min} (input_size + h_train + test_size) "
        )

        super().__init__(detail=detail_msg, **data)

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
        default=PatienceStepsError()
    )