from pydantic import Field

from src.core.application.building_model.schemas.lstm import LstmParams, LstmFitRequest
from src.core.domain import ForecastResult_V2


class LstmParams_V2(LstmParams):
    input_size: int = Field(ge=1, default=8, title='Размер входного слоя')
    output_size: int = Field(ge=1, default=2, title='Размер выходного слоя')


class LstmFitRequest_V2(LstmFitRequest):
    hyperparameters: LstmParams_V2 = Field(title="Параметры модели NHiTS")


class LstmFitResult_V2(ForecastResult_V2):
    pass
