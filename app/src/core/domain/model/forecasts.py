from typing import Optional

from pydantic import BaseModel, Field

from src.core.domain.timeseries.timeseries import Timeseries


class Forecasts(BaseModel):
    train_predict: Timeseries = Field(
        default=Timeseries(name="Прогноз на обучающей выборке"),
        title="Прогноз на обучающей выборке"
    )
    validation_predict: Optional[Timeseries] = Field(
        default=Timeseries(name="Прогноз на валидационной выборке"),
        title="Прогноз на валидационной выборке"
    )
    test_predict: Optional[Timeseries] = Field(
        default=Timeseries(name="Прогноз на тестовой выборке"),
        title="Прогноз на тестовой выборке"
    )
    forecast: Optional[Timeseries] = Field(
        default=Timeseries(name="Прогноз дальше известных данных"),
        title="Прогноз дальше известных данных"
    )

