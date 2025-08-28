from typing import Optional, List
from pydantic import BaseModel, Field

from src.core.domain import Timeseries, Forecasts, ForecastAnalysis
from src.core.domain.timeseries.timeseries import gen_values, gen_dates


class BreuschGodfreyData(ForecastAnalysis):
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
