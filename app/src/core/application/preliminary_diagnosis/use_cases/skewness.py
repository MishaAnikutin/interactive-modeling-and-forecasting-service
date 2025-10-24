from src.core.application.preliminary_diagnosis.schemas.skewness import SkewnessParams, SkewnessResult
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
import numpy as np
from scipy.stats import skew


class SkewnessUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: SkewnessParams) -> SkewnessResult:
        sample = np.array(request.timeseries.values)
        sample = sample[~np.isnan(sample)]
        skewness = skew(sample, bias=False)

        return SkewnessResult(value=round(skewness, 4))
