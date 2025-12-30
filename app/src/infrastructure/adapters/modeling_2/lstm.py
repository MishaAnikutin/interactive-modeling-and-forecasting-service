from neuralforecast.models import LSTM

from src.core.application.building_model.schemas.lstm_v2 import LstmParams_V2, LstmFitResult_V2
from src.infrastructure.adapters.modeling_2.base import BaseNeuralForecast


class LstmAdapter_V2(BaseNeuralForecast[LstmParams_V2]):
    model_name = "LSTM"
    model_class = LSTM
    result_class = LstmFitResult_V2

