from statsmodels.stats.stattools import omni_normtest

from src.core.application.model_diagnosis.schemas.common import StatTestResult
from src.core.application.model_diagnosis.schemas.omnibus import OmnibusRequest
from src.core.application.model_diagnosis.use_cases.interface import ResidAnalysisInterface
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter
from src.shared.full_predict import get_full_predict
from src.shared.get_residuals import get_residuals


class OmnibusUC(ResidAnalysisInterface):
    def __init__(
            self,
            ts_aligner: TimeseriesAlignment,
            ts_adapter: PandasTimeseriesAdapter,
    ):
        super().__init__(ts_aligner, ts_adapter)

    def execute(self, request: OmnibusRequest) -> StatTestResult:
        target, _ = self._aligned_data(request.data.target, exog=None)
        residuals = get_residuals(
            y_true=target,
            y_pred=get_full_predict(target, request.data.forecasts)
        )
        stat_value, p_value = omni_normtest(residuals)
        return StatTestResult(
            p_value=p_value,
            stat_value=stat_value,
        )
