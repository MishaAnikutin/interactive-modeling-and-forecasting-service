from typing import Annotated, Union, Literal

from pydantic import Field, BaseModel

from src.core.application.building_model.errors.nhits import HorizonValidationError, ValSizeError, PatienceStepsError


class HiddenSizeError(BaseModel):
    type: Literal["hidden_size"] = "hidden_size"
    detail: str = Field(
        default=(
            "Hidden size must be greater than proj size."
            "Proj size is equal to 1 if recurrent set to True and 0 otherwise."
        ),
        title="Описание ошибки"
    )

PydanticValidationErrorType = Annotated[
    Union[
        HiddenSizeError
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
        LstmTrainSizeError2
    ],
    Field(discriminator="type")
]

class LstmFitValidationError(BaseModel):
    msg: FitValidationErrorType = Field(
        title="Описание ошибки",
        default=HorizonValidationError()
    )