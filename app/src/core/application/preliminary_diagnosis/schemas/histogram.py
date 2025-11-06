from pydantic import BaseModel

from src.core.domain import Timeseries
from src.core.domain.distributions import EstimateHistogramParams


class HistogramRequest(BaseModel):
    timeseries: Timeseries
    histogram_params: EstimateHistogramParams