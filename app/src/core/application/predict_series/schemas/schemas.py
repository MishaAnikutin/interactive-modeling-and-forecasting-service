from pydantic import BaseModel, Field

from src.core.domain import Timeseries
from src.core.domain.model.model_data import ModelData


class PredictRequest(BaseModel):
    model_weight: str = Field(
        default="example",
        title="Веса модели",
        description="Веса модели в UTF-8 формате после преобразования в pickle"
    )
    forecast_steps: int = Field(
        ge=0,
        le=10000,
        default=0,
        title="Число шагов прогноза"
    )
    model_data: ModelData = Field(default=ModelData())


class PredictResponse(BaseModel):
    in_sample_predict: Timeseries
    out_of_sample_predict: Timeseries
