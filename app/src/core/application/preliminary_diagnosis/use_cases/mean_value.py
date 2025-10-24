from src.core.application.preliminary_diagnosis.schemas.mean_value import MeanResult, MeanParams
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
import numpy as np

class MeanUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: MeanParams) -> MeanResult:
        mean = np.mean(request.timeseries.values)
        return MeanResult(value=round(mean, 4))
