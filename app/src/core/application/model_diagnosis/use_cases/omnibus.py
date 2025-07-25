from statsmodels.stats.stattools import omni_normtest

from src.core.application.model_diagnosis.schemas.common import StatTestResult
from src.core.application.model_diagnosis.schemas.omnibus import OmnibusRequest
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
from src.shared.full_predict import get_full_predict
from src.shared.get_residuals import get_residuals


class OmnibusUC:
    def __init__(
            self,
            ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: OmnibusRequest) -> StatTestResult:
        residuals = get_residuals(
            y_true=request.data.ts,
            y_pred=get_full_predict(request.data.ts, request.data.forecasts)
        )
        p_value, stat_value = omni_normtest(residuals)
        return StatTestResult(
            p_value=p_value,
            stat_value=stat_value,
        )