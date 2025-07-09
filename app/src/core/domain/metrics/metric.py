from typing import Optional

from pydantic import BaseModel


class Metric(BaseModel):
    type: str
    value: Optional[float]
