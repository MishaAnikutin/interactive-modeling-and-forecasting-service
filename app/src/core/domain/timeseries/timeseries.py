from datetime import datetime, timedelta
from typing import Optional
import numpy as np
from pydantic import BaseModel, Field

n = 120

def gen_dates():
    dates =[datetime(2023, 1, 1) + timedelta(days=i*30) for i in range(n)]
    return dates

def gen_values():
    pageviews = np.random.randint(1000, 5000, size=n)
    pageviews = pageviews + np.sin(np.arange(n) * 2 * np.pi / 7) * 500  # Добавляем недельную сезонность
    pageviews = pageviews + np.sin(np.arange(n) * 2 * np.pi / 365) * 1000  # Добавляем годовую сезонность
    pageviews = pageviews + np.arange(n) * 2  # Добавляем тренд
    return pageviews.tolist()

class Timeseries(BaseModel):
    name: str = Field(title="Название ряда", default="IPP")
    dates: list[datetime] = Field(
        title="Даты",
        default=gen_dates()
    )
    values: list[Optional[float]] = Field(
        title="Значения",
        default=gen_values()
    )
