from pydantic import Field, BaseModel

from .. import Forecasts, Timeseries


class ForecastAnalysis(BaseModel):
    """Прогнозы и исходные данные"""
    forecasts: Forecasts = Field(
        title="Прогнозы",
        description="Даты объединенного прогноза должны совпадать с датами исторических данных."
    )

    target: Timeseries = Field(
        title="Исходные данные",
        default=Timeseries(name="Исходные данные")
    )