from src.core.application.preliminary_diagnosis.schemas.kurtosis import KurtosisParams, KurtosisResult
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
import numpy as np
from scipy.stats import kurtosis


class KurtosisUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: KurtosisParams) -> KurtosisResult:
        sample = np.array(request.timeseries.values)
        sample = sample[~np.isnan(sample)]
        kurtosis_value = kurtosis(sample, bias=False, fisher=True)
        return KurtosisResult(value=round(kurtosis_value, 4))
