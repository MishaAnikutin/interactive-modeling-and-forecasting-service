from statsmodels.stats.diagnostic import kstest_normal

from src.core.application.model_diagnosis.schemas.common import StatTestResult
from src.core.application.model_diagnosis.schemas.kstest import KolmogorovRequest
from src.core.application.model_diagnosis.use_cases.interface import ResidAnalysisInterface
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter

from src.shared.full_predict import get_full_predict
from src.shared.get_residuals import get_residuals


class KolmogorovUC(ResidAnalysisInterface):
    def __init__(
            self,
            ts_aligner: TimeseriesAlignment,
            ts_adapter: PandasTimeseriesAdapter,
    ):
        super().__init__(ts_aligner, ts_adapter)

    def execute(self, request: KolmogorovRequest) -> StatTestResult:
        target, _ = self._aligned_data(request.data.target, request.data.exog)
        residuals = get_residuals(
            y_true=target,
            y_pred=get_full_predict(target, request.data.forecasts)
        )
        p_value, stat_value = kstest_normal(residuals)
        return StatTestResult(
            p_value=p_value,
            stat_value=stat_value,
        )