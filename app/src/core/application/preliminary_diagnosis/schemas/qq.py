from pydantic import BaseModel, Field
from src.core.domain import Timeseries


class QQParams(BaseModel):
    timeseries: Timeseries


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
