from statsmodels.stats.diagnostic import acorr_lm

from src.core.application.model_diagnosis.schemas.arch import ArchRequest
from src.core.application.model_diagnosis.schemas.common import DiagnosticsResult
from src.core.application.model_diagnosis.use_cases.interface import ResidAnalysisInterface
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter
from src.shared.full_predict import get_full_predict
from src.shared.get_residuals import get_residuals


class ArchUC(ResidAnalysisInterface):
    def __init__(
            self,
            ts_aligner: TimeseriesAlignment,
            ts_adapter: PandasTimeseriesAdapter,
    ):
        super().__init__(ts_aligner, ts_adapter)

    def execute(self, request: ArchRequest) -> DiagnosticsResult:
        target, _ = self._aligned_data(request.data.target, request.data.exog)
        residuals = get_residuals(
            y_true=target,
            y_pred=get_full_predict(target, request.data.forecasts)
        )
        lmval, lmpval, fval, fpval = acorr_lm(residuals ** 2, ddof=2)
        return DiagnosticsResult(
            lmval=lmval,
            lmpval=lmpval,
            fval=fval,
            fpval=fpval,
        )