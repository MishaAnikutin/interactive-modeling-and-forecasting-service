from typing import List, Optional

from pydantic import BaseModel, Field

from .schemas import Forecasts, ModelMetrics, FitParams
from src.shared.schemas import Coefficient, Timeseries


class ArimaxParams(BaseModel):
    p: int = Field(gte=0)
    d: int = Field(gte=0)
    q: int = Field(gte=0)


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
