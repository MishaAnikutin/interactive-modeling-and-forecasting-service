from pydantic import BaseModel


class Metric(BaseModel):
    type: str
    value: float
