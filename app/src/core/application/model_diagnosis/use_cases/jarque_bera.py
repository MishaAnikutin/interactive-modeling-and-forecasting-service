from statsmodels.stats.stattools import jarque_bera

from src.core.application.model_diagnosis.schemas.jarque_bera import JarqueBeraRequest, JarqueBeraResult
from src.core.application.model_diagnosis.use_cases.interface import ResidAnalysisInterface
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter
from src.shared.full_predict import get_full_predict
from src.shared.get_residuals import get_residuals


class JarqueBeraUC(ResidAnalysisInterface):
    def __init__(
            self,
            ts_aligner: TimeseriesAlignment,
            ts_adapter: PandasTimeseriesAdapter,
    ):
        super().__init__(ts_aligner, ts_adapter)

    def execute(self, request: JarqueBeraRequest) -> JarqueBeraResult:
        target, _ = self._aligned_data(request.data.target, exog=None)
        residuals = get_residuals(
            y_true=target,
            y_pred=get_full_predict(target, request.data.forecasts)
        )
        stat_value, p_value, skew, kurtosis = jarque_bera(residuals)
        return JarqueBeraResult(
            p_value=p_value,
            stat_value=stat_value,
            skew=skew,
            kurtosis=kurtosis,
        )
