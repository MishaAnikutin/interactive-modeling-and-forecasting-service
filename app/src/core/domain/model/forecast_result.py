from pydantic import BaseModel, Field

from src.core.domain import Forecasts, ModelMetrics


class ForecastResult(BaseModel):
    forecasts: Forecasts = Field(title="Прогнозы")
    model_metrics: ModelMetrics = Field(title="Метрики модели")
