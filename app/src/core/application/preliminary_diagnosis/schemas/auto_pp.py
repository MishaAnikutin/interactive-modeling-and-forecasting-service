from pydantic import BaseModel, Field

from src.core.domain import Timeseries
from src.core.domain.distributions import SelectDistributionMethod


class AutoPPRequest(BaseModel):
    timeseries: Timeseries
    method: SelectDistributionMethod = Field(description="Метод нахождения распределения")