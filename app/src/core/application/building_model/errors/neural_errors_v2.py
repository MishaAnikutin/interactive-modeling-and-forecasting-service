from typing import Annotated, Union

from pydantic import Field, BaseModel

from src.infrastructure.adapters.modeling_2.neural_models_errors import PatienceStepsError, ValSizeError, \
    TrainSizeError, LSTM_GRU_TrainSizeError
from src.core.application.building_model.errors.alignment import NotEqualToExpectedError, \
    NotConstantFreqError, NotSupportedFreqError, NotLastDayOfMonthError, EmptyError, NoDataAfterAlignmentError, \
    BoundariesError

FitValidationErrorType = Annotated[
    Union[
        ValSizeError,
        PatienceStepsError,
        TrainSizeError,
        LSTM_GRU_TrainSizeError,
        NoDataAfterAlignmentError,
        NotEqualToExpectedError,
        NotConstantFreqError,
        NotSupportedFreqError,
        NotLastDayOfMonthError,
        EmptyError
    ],
    Field(discriminator="type")
]

class NeuralModelsFitValidationError(BaseModel):
    msg: FitValidationErrorType = Field(
        title="Описание ошибки",
        default=PatienceStepsError()
    )