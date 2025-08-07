from typing import List, Optional, Any

from pydantic import BaseModel, Field

from src.core.domain import Timeseries, FitParams, Forecasts, Coefficient, ModelMetrics
from src.core.domain.model.model_data import ModelData


class ArimaxParams(BaseModel):
    p: int = Field(default=0, ge=0, le=10000)
    d: int = Field(default=0, ge=0, le=10000)
    q: int = Field(default=0, ge=0, le=10000)


class ArimaxFitRequest(BaseModel):
    model_data: ModelData = Field(default=ModelData())
    hyperparameters: ArimaxParams = Field(title='Параметры модели ARIMAX')
    fit_params: FitParams = Field(
        default=FitParams(),
        title="Общие параметры обучения",
        description="train_boundary должна быть раньше val_boundary ",
    )


class ArimaxFitResult(BaseModel):
    forecasts: Forecasts = Field(title="Прогнозы")
    coefficients: List[Coefficient] = Field(title="Список коэффициентов")
    model_metrics: ModelMetrics = Field(title="Метрики модели")


class ArimaxFitResponse(BaseModel):
    fit_result: ArimaxFitResult
    model_weight: Any = Field(title='веса модели, которые нужно сериализовать и сохранить в S3')
