from pydantic import Field

from src.core.application.building_model.schemas import NhitsFitRequest
from src.core.application.building_model.schemas.nhits import NhitsParams
from src.core.domain import ForecastResult_V2


class NhitsParams_V2(NhitsParams):
    input_size: int = Field(default=8, title='Размер входного слоя')
    output_size: int = Field(default=2, title='Размер выходного слоя')


class NhitsFitRequest_V2(NhitsFitRequest):
    hyperparameters: NhitsParams_V2 = Field(title="Параметры модели NHiTS")


class NhitsFitResult_V2(ForecastResult_V2):
    pass
