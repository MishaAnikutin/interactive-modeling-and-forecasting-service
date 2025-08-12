import json

from pydantic import BaseModel, Field, model_validator

from src.core.application.building_model.schemas.gru import GruParams
from src.core.domain import Timeseries
from src.core.domain.model.model_data import ModelData


class PredictArimaxRequest(BaseModel):
    predict_params: ModelData = Field(default=ModelData(), title="Новые данные для прогноза")
    forecast_steps: int = Field(
        gt=0,
        le=10000,
        default=1,
        title="Число шагов прогноза"
    )

    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


class PredictGruRequest(BaseModel):
    predict_params: ModelData = Field(default=ModelData(), title="Новые данные для прогноза")
    forecast_steps: int = Field(
        gt=0,
        le=10000,
        default=1,
        title="Число шагов прогноза"
    )
    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

class PredictResponse(BaseModel):
    in_sample_predict: Timeseries
    out_of_sample_predict: Timeseries
