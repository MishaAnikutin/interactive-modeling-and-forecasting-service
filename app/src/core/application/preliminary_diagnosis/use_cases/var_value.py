from src.core.application.preliminary_diagnosis.schemas.var_value import VarianceResult, VarianceParams
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
import numpy as np

class VarianceUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: VarianceParams) -> VarianceResult:
        var = np.var(request.timeseries.values)
        return VarianceResult(value=round(var, 4))
