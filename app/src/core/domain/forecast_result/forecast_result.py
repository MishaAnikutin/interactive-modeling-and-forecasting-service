from typing import List

from pydantic import BaseModel, Field

from .. import Forecasts, ModelMetrics, Metric, Timeseries


class ForecastResult(BaseModel):
    forecasts: Forecasts = Field(title="Прогнозы")
    model_metrics: ModelMetrics = Field(title="Метрики модели")

class ForecastResult_V2(ForecastResult):
    forecasts: List[Timeseries] = Field(title="Прогнозы на скользящих окнах")
    model_metrics: List[List[Metric]] = Field(title="Метрики на скользящих окнах")