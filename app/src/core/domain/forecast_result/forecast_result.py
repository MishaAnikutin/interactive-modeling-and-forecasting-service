from typing import List

from pydantic import BaseModel, Field

from .. import Forecasts, ModelMetrics, Timeseries


class ForecastResult(BaseModel):
    forecasts: Forecasts = Field(title="Прогнозы")
    model_metrics: ModelMetrics = Field(title="Метрики модели")

class ForecastResult_V2(BaseModel):
    forecasts: List[Timeseries] = Field(title="Прогнозы на скользящих окнах")
    best_forecast: Timeseries = Field(title="Прогноз составленный из первых точек прогнозов на окнах")
    best_forecast_metrics: ModelMetrics = Field(title="Метрики на прогнозе из первых точек")