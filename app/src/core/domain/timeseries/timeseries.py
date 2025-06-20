from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Timeseries(BaseModel):
    name: str
    dates: list[datetime]
    values: list[Optional[float]]
