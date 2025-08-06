from pydantic import BaseModel, Field

from src.core.application.building_model.schemas.gru import GruParams
from src.core.domain import Timeseries
from src.core.domain.model.model_data import ModelData


class PredictParams(BaseModel):
    """Данные для получения прогнозов, зная веса."""
    model_weight: str = Field(
        default="example",
        title="Веса модели",
        description="Веса модели в UTF-8 формате после преобразования в pickle"
    )
    model_data: ModelData = Field(default=ModelData())


class PredictArimaxRequest(BaseModel):
    predict_params: PredictParams = Field(default=PredictParams())
    forecast_steps: int = Field(
        ge=0,
        le=10000,
        default=0,
        title="Число шагов прогноза"
    )


class PredictGruRequest(BaseModel):
    predict_params: PredictParams = Field(default=PredictParams())
    gru_params: GruParams = Field(default=GruParams())


class PredictResponse(BaseModel):
    in_sample_predict: Timeseries
    out_of_sample_predict: Timeseries
