from statsmodels.stats.diagnostic import acorr_lm

from src.core.application.model_diagnosis.schemas.arch import ArchOrLmRequest
from src.core.application.model_diagnosis.schemas.common import DiagnosticsResult
from src.core.application.model_diagnosis.use_cases.interface import ResidAnalysisInterface
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter
from src.shared.full_predict import get_full_predict
from src.shared.get_residuals import get_residuals


class LmUC(ResidAnalysisInterface):
    def __init__(
            self,
            ts_aligner: TimeseriesAlignment,
            ts_adapter: PandasTimeseriesAdapter,
    ):
        super().__init__(ts_aligner, ts_adapter)

    def execute(self, request: ArchOrLmRequest) -> DiagnosticsResult:
        target, _ = self._aligned_data(request.data.target, exog=None)
        residuals = get_residuals(
            y_true=target,
            y_pred=get_full_predict(target, request.data.forecasts)
        )
        lmval, lmpval, fval, fpval = acorr_lm(
            resid=residuals,
            ddof=request.ddof,
            nlags=request.nlags,
            cov_type=request.cov_type,
            period=request.period
        )
        return DiagnosticsResult(
            lmval=lmval,
            lmpval=lmpval,
            fval=fval,
            fpval=fpval,
        )