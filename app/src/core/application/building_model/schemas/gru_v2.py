from pydantic import Field

from src.core.application.building_model.schemas.gru import GruParams, GruFitRequest
from src.core.domain import ForecastResult_V2


class GruParams_V2(GruParams):
    input_size: int = Field(default=8, title='Размер входного слоя')
    output_size: int = Field(default=2, title='Размер выходного слоя')


class GruFitRequest_V2(GruFitRequest):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               GruFitRequest):
    hyperparameters: GruParams_V2 = Field(title="Параметры модели NHiTS")


class GruFitResult_V2(ForecastResult_V2):
    pass
