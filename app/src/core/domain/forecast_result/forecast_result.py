from typing import List, Optional

from pydantic import BaseModel, Field

from .. import Forecasts, ModelMetrics, Timeseries


class ForecastResult(BaseModel):
    forecasts: Forecasts = Field(title="Прогнозы")
    model_metrics: ModelMetrics = Field(title="Метрики модели")


class WindowsForecast(BaseModel):
    train_forecasts: List[Timeseries] = Field(title="Прогнозы на скользящих окнах внутри обучающей выборки")
    val_forecasts: List[Timeseries] = Field(title="Прогнозы на скользящих окнах внутри валидационной выборки")
    test_forecasts: List[Timeseries] = Field(title="Прогнозы на скользящих окнах внутри тестовой выборки")
    out_of_sample_forecasts: List[Timeseries] = Field(title="Прогнозы на скользящих окнах дальше известных данных")


class ForecastResult_V2(BaseModel):
    forecasts: WindowsForecast = Field(title="Прогнозы на скользящих окнах")
    best_forecast: Timeseries = Field(title="Прогноз составленный из первых точек прогнозов на окнах")
    best_forecast_metrics: ModelMetrics = Field(title="Метрики на прогнозе из первых точек")