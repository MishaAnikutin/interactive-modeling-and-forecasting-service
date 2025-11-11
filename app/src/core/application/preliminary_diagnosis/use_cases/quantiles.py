from src.core.application.preliminary_diagnosis.schemas.quantiles import QuantilesParams, QuantilesResult
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
import numpy as np

class QuantilesUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: QuantilesParams) -> QuantilesResult:
        sample = np.array(request.timeseries.values)
        sample = sample[~np.isnan(sample)]
        quantiles = dict(
            q_0=0,
            q_1=1,
            q_5=5,
            q_25=25,
            q_50=50,
            q_75=75,
            q_95=95,
            q_99=99,
            q_100=100
        )

        quantiles_values = {}
        for name, q in quantiles.items():
            quantiles_values[name] = round(np.percentile(sample, q), 4)
        return QuantilesResult(**quantiles_values)
