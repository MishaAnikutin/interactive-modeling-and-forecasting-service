from pydantic import BaseModel, Field

from src.core.application.preliminary_diagnosis.schemas.pp_plot import PPResult
from src.core.domain import Timeseries


class AutoPPRequest(BaseModel):
    timeseries: Timeseries


class AutoPPResult(PPResult):
    dist_name: str = Field(..., title='Название распределения, которое получено в подборе')