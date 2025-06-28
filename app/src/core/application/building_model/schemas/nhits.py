from typing import List, Optional

from pydantic import BaseModel, Field

from src.core.domain import (
    Timeseries,
    FitParams,
    Forecasts,
    ModelMetrics,
)


class NhitsParams(BaseModel):
    """
    Гиперпараметры модели NHiTS.

    Можно уточнять/расширять по необходимости.
    """
    h: int = Field(..., gt=0, description="Горизонт прогноза")
    input_size: int = Field(..., gt=0, description="Длина входного окна")
    max_steps: int = Field(default=500, gt=0, description="Максимум итераций обучения")
    early_stop_patience_steps: int = Field(default=10, ge=0, description="Patience для early-stopping")
    learning_rate: float = Field(default=1e-3, gt=0, description="Шаг обучения")
    scaler_type: str = Field(default="robust", description="Тип скейлера")


class NhitsFitRequest(BaseModel):
    dependent_variables: Timeseries
    explanatory_variables: Optional[List[Timeseries]]
    hyperparameters: NhitsParams
    fit_params: FitParams


class NhitsFitResult(BaseModel):
    forecasts: Forecasts          # прогнозы train/val/test
    model_metrics: ModelMetrics   # рассчитанные метрики
    weight_path: str              # путь к сохранённым весам
    model_id: str                 # идентификатор модели