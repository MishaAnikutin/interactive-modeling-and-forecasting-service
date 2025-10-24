from src.core.application.preliminary_diagnosis.schemas.cv_value import VariationCoeffResult, VariationCoeffParams
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
import numpy as np

class VariationCoeffUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: VariationCoeffParams) -> VariationCoeffResult:
        mean = np.mean(request.timeseries.values)
        std = np.std(request.timeseries.values, ddof=1)
        return VariationCoeffResult(value=round(100 * std / mean, 4))
