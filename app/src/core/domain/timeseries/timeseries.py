from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel, Field


class Timeseries(BaseModel):
    name: str = Field(title="Название ряда", default="IPP")
    dates: list[datetime] = Field(
        title="Даты",
        default=[datetime(2023, 1, 1) + timedelta(days=i*30) for i in range(10)]
    )
    values: list[Optional[float]] = Field(
        title="Значения",
        default=[float(i * 10 + 5) for i in range(10)]
    )
