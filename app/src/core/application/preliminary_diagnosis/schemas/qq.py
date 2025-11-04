from pydantic import BaseModel, Field
from src.core.domain import Timeseries
from src.core.domain.distributions import Distribution


class QQParams(BaseModel):
    timeseries: Timeseries
    theoretical_dist: Distribution


class QQResult(BaseModel):
    data_values: list[float] = Field(
        ...,
        title="Значения данных",
        description="Эмпирические значения выборки."
    )
    normal_values: list[float] = Field(
        ...,
        title="Нормальные значения",
        description="Теоретические значения нормального распределения."
    )
