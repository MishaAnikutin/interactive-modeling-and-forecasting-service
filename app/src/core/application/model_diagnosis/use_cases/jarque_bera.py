from statsmodels.stats.stattools import jarque_bera

from src.core.application.model_diagnosis.schemas.jarque_bera import JarqueBeraRequest, JarqueBeraResult
from src.shared.full_predict import get_full_predict
from src.shared.get_residuals import get_residuals


class JarqueBeraUC:
    def execute(self, request: JarqueBeraRequest) -> JarqueBeraResult:
        residuals = get_residuals(
            y_true=request.data.ts,
            y_pred=get_full_predict(request.data.ts, request.data.forecasts)
        )
        p_value, stat_value, skew, kurtosis = jarque_bera(residuals)
        return JarqueBeraResult(
            p_value=p_value,
            stat_value=stat_value,
            skew=skew,
            kurtosis=kurtosis,
        )