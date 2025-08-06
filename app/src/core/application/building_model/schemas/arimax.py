from typing import List, Optional

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


class ArimaxFitResult(BaseModel):
    forecasts: Forecasts = Field(title="Прогнозы")
    coefficients: List[Coefficient] = Field(title="Список коэффициентов")
    model_metrics: ModelMetrics = Field(title="Метрики модели")


class ArimaxFitResponse(BaseModel):
    fit_result: ArimaxFitResult

    # FIXME: в этом костыле тут не просто байты pickle а еще в utf-8
    serialized_model_weight: str = Field(title='Сериализованные веса модели')
