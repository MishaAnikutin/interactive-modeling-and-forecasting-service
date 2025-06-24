from typing import List, Optional

from pydantic import BaseModel, Field

from src.core.domain import Timeseries, FitParams, Forecasts, Coefficient, ModelMetrics


class ArimaxParams(BaseModel):
    p: int = Field(default=0, ge=0)
    d: int = Field(default=0, ge=0)
    q: int = Field(default=0, ge=0)


class ArimaxFitRequest(BaseModel):
    dependent_variables: Timeseries
    explanatory_variables: Optional[List[Timeseries]]
    hyperparameters: ArimaxParams
    fit_params: FitParams


class ArimaxFitResult(BaseModel):
    forecasts: Forecasts
    coefficients: List[Coefficient]
    model_metrics: ModelMetrics
    weight_path: str
    model_id: str
