import json

from pydantic import BaseModel, Field, model_validator

from src.core.domain import FitParams
from src.core.domain.model.model_data import ModelData


class PredictRequest(BaseModel):
    model_data: ModelData = Field(default=ModelData(), title="Новые данные для прогноза")
    fit_params: FitParams = Field(
        title="Общие параметры обучения",
    )

    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

