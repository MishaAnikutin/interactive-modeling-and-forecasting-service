from datetime import date  # Изменено: добавлен date
from typing import Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from src.core.domain.timeseries.data_frequency import DataFrequency
from src.shared.utils import validate_float_param

def gen_dates(num: int) -> list[date]:
    """Возвращает n дат типа «month ending», начиная с 31-01-2023."""
    dates_idx = pd.date_range(start="2023-01-31", periods=num, freq="ME")
    return [d.date() for d in dates_idx]

def gen_values(num: int):
    pageviews = np.random.randint(1000, 5000, size=num)
    pageviews = pageviews + np.sin(np.arange(num) * 2 * np.pi / 7) * 500
    pageviews = pageviews + np.sin(np.arange(num) * 2 * np.pi / 365) * 1000
    pageviews = pageviews + np.arange(num) * 2
    return pageviews.tolist()

class Timeseries(BaseModel):
    name: str = Field(title="Название ряда", default="IPP")
    dates: list[date] = Field(
        title="Даты",
        default=gen_dates(100)
    )
    values: list[Optional[float]] = Field(
        title="Значения",
        default=gen_values(100)
    )

    data_frequency: DataFrequency = Field(
        default=DataFrequency.month,
        title="Частотность ряда"
    )

    @model_validator(mode="after")
    def validate_value(self):
        self.values = [validate_float_param(value) for value in self.values]
        return self
