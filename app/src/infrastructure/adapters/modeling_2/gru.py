from neuralforecast.models import GRU

from src.core.application.building_model.schemas.gru_v2 import GruParams_V2, GruFitResult_V2
from src.infrastructure.adapters.modeling_2.base import BaseNeuralForecast


class GruAdapter_V2(BaseNeuralForecast[GruParams_V2]):
    model_name = "GRU"
    model_class = GRU
    result_class = GruFitResult_V2

