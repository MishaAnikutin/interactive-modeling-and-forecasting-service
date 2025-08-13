from typing import List, Optional, Any

from pydantic import BaseModel, Field, model_validator

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

    @model_validator(mode='after')
    def validate_params(self):
        start_date = min(self.model_data.dependent_variables.dates)
        end_date = max(self.model_data.dependent_variables.dates)

        if not (end_date > self.fit_params.val_boundary > self.fit_params.train_boundary > start_date):
            raise ValueError('Границы выборок должны быть внутри датасета')

        return self


class ArimaxFitResult(BaseModel):
    forecasts: Forecasts = Field(title="Прогнозы")
    coefficients: List[Coefficient] = Field(title="Список коэффициентов")
    model_metrics: ModelMetrics = Field(title="Метрики модели")

