from pydantic import Field, BaseModel

from src.core.application.preliminary_diagnosis.schemas.qq import QQResult
from src.core.domain import Timeseries


class AutoQQRequest(BaseModel):
    timeseries: Timeseries


class AutoQQResult(QQResult):
    dist_name: str = Field(..., title='Название распределения, которое получено в подборе')