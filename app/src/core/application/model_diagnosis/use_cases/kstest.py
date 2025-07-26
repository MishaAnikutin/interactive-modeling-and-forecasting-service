from statsmodels.stats.diagnostic import kstest_normal

from src.core.application.model_diagnosis.schemas.common import StatTestResult
from src.core.application.model_diagnosis.schemas.kstest import KolmogorovRequest

from src.shared.full_predict import get_full_predict
from src.shared.get_residuals import get_residuals


class KolmogorovUC:
    def execute(self, request: KolmogorovRequest) -> StatTestResult:
        residuals = get_residuals(
            y_true=request.data.ts,
            y_pred=get_full_predict(request.data.ts, request.data.forecasts)
        )
        p_value, stat_value = kstest_normal(residuals)
        return StatTestResult(
            p_value=p_value,
            stat_value=stat_value,
        )