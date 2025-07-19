from datetime import date  # Изменено: добавлен date
from typing import Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from src.core.domain.timeseries.data_frequency import DataFrequency

n = 100

def gen_dates() -> list[date]:
    """Возвращает n дат типа «month ending», начиная с 31-01-2023."""
    dates_idx = pd.date_range(start="2023-01-31", periods=n, freq="ME")
    return [d.date() for d in dates_idx]

def gen_values():
    pageviews = np.random.randint(1000, 5000, size=n)
    pageviews = pageviews + np.sin(np.arange(n) * 2 * np.pi / 7) * 500
    pageviews = pageviews + np.sin(np.arange(n) * 2 * np.pi / 365) * 1000
    pageviews = pageviews + np.arange(n) * 2
    return pageviews.tolist()

class Timeseries(BaseModel):
    name: str = Field(title="Название ряда", default="IPP")
    dates: list[date] = Field(
        title="Даты",
        default=gen_dates()
    )
    values: list[Optional[float]] = Field(
        title="Значения",
        default=gen_values()
    )

    data_frequency: DataFrequency = Field(
        default=DataFrequency.month,
        title="Частотность ряда"
    )
