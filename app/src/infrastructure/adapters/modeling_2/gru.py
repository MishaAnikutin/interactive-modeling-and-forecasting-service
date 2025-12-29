from typing import Type

from neuralforecast.models import GRU

from src.core.application.building_model.schemas.gru_v2 import GruParams_V2
from src.infrastructure.adapters.modeling_2.base import BaseNeuralForecast, TResult


class GruAdapter_V2(BaseNeuralForecast[GruParams_V2]):
    @property
    def result_class(self) -> Type[TResult]:
        return None

    model_name = "GRU"
    model_class = GRU

