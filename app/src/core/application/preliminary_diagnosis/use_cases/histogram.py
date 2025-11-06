from src.core.application.preliminary_diagnosis.schemas.histogram import HistogramRequest
from src.core.domain.distributions import Histogram
from src.infrastructure.adapters.distributions import HistogramEstimator


class HistogramUC:
    def __init__(
            self,
            histogram_estimator: HistogramEstimator
    ):
        self._histogram_estimator = histogram_estimator

    def execute(self, request: HistogramRequest) -> Histogram:
        histogram: Histogram = self._histogram_estimator.eval(
            values=request.timeseries.values,
            bins=request.histogram_params.bins,
            is_density=request.histogram_params.is_density
        )
        return histogram
