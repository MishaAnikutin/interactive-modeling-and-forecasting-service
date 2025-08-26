from typing import Optional, List
from pydantic import BaseModel, Field

from src.core.domain import Timeseries, Forecasts
from src.core.domain.timeseries.timeseries import gen_values, gen_dates


class BreuschGodfreyData(BaseModel):
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

    exog: Optional[List[Timeseries]] = Field(
        default=None,
        title="Экзогенные переменные (опционально)"
    )


class BreuschGodfreyRequest(BaseModel):
    data: BreuschGodfreyData = Field(
        title="Прогнозы и исходные данные"
    )

    nlags: Optional[int] = Field(
        default=None,
        ge=1, le=10000,
        title="Число лагов"
    )
