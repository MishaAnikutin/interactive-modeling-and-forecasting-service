from pydantic import BaseModel


class TestMetric(BaseModel):
    stat_value: float
    p_value: float
