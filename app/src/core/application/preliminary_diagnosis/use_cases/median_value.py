from src.core.application.preliminary_diagnosis.schemas.median_value import MedianParams, MedianResult
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
import numpy as np

class MedianUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: MedianParams) -> MedianResult:
        mean = np.median(request.timeseries.values)
        return MedianResult(value=round(mean, 4))
