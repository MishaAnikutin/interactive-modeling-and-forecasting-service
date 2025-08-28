from typing import Optional

from pydantic import BaseModel, Field

from src.core.domain.timeseries.timeseries import Timeseries, gen_values, gen_dates


class Forecasts(BaseModel):
    train_predict: Timeseries = Field(
        default=Timeseries(
            values=gen_values(110)[:70],
            dates=gen_dates(110)[:70],
            name="Прогноз на обучающей выборке"
        ),
        title="Прогноз на обучающей выборке"
    )
    validation_predict: Optional[Timeseries] = Field(
        default=Timeseries(
            values=gen_values(110)[70:90],
            dates=gen_dates(110)[70:90],
            name="Прогноз на валидационной выборке"
        ),
        title="Прогноз на валидационной выборке"
    )
    test_predict: Optional[Timeseries] = Field(
        default=Timeseries(
            values=gen_values(110)[90:100],
            dates=gen_dates(110)[90:100],
            name="Прогноз на тестовой выборке"
        ),
        title="Прогноз на тестовой выборке"
    )
    forecast: Optional[Timeseries] = Field(
        default=Timeseries(
            values=gen_values(110)[100:],
            dates=gen_dates(110)[100:],
            name="Прогноз на вне выборки"
        ),
        title="Прогноз дальше известных данных"
    )

