from typing import Type

from neuralforecast.models import LSTM

from src.core.application.building_model.schemas.lstm_v2 import LstmParams_V2
from src.infrastructure.adapters.modeling_2.base import BaseNeuralForecast, TResult


class LstmAdapter_V2(BaseNeuralForecast[LstmParams_V2]):
    @property
    def result_class(self) -> Type[TResult]:
        return None

    model_name = "LSTM"
    model_class = LSTM

