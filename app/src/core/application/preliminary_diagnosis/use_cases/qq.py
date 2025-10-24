from src.core.application.preliminary_diagnosis.schemas.qq import QQResult, QQParams
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
import numpy as np
from scipy import stats


class QQplotUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: QQParams) -> QQResult:
        sample = np.array(request.timeseries.values)
        sample = sample[~np.isnan(sample)]

        sample_sorted = np.sort(sample)
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(sample_sorted)),
            loc=np.mean(sample_sorted),
            scale=np.std(sample_sorted, ddof=1)
        )

        return QQResult(
            normal_values=theoretical_quantiles.tolist(),
            data_values=sample_sorted.tolist()
        )
