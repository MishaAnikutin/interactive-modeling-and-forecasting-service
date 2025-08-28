from pydantic import Field, BaseModel

from .. import Forecasts, Timeseries
from ..timeseries.timeseries import gen_dates, gen_values


class ForecastAnalysis(BaseModel):
    """Прогнозы и исходные данные"""
    forecasts: Forecasts = Field(
        default=Forecasts(
            train_predict=Timeseries(
                values=gen_values(110)[:70],
                dates=gen_dates(110)[:70],
                name="Прогноз на обучающей выборке"
            ),
            validation_predict=Timeseries(
                values=gen_values(110)[70:90],
                dates=gen_dates(110)[70:90],
                name="Прогноз на валидационной выборке"
            ),
            test_predict=Timeseries(
                values=gen_values(110)[90:100],
                dates=gen_dates(110)[90:100],
                name="Прогноз на тестовой выборке"
            ),
            forecast=Timeseries(
                values=gen_values(110)[100:],
                dates=gen_dates(110)[100:],
                name="Прогноз на вне выборки"
            ),
        ),
        title="Прогнозы",
        description="Даты объединенного прогноза должны совпадать с датами исторических данных."
    )

    target: Timeseries = Field(
        title="Исходные данные",
        default=Timeseries(name="Исходные данные")
    )