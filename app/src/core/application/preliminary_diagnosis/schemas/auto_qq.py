from pydantic import Field, BaseModel

from src.core.domain import Timeseries
from src.core.domain.distributions import SelectDistributionMethod


class AutoQQRequest(BaseModel):
    timeseries: Timeseries
    method: SelectDistributionMethod = Field(description="Метод нахождения распределения")