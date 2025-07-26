from statsmodels.stats.diagnostic import acorr_lm

from src.core.application.model_diagnosis.schemas.arch import ArchRequest, ArchResult
from src.shared.full_predict import get_full_predict
from src.shared.get_residuals import get_residuals


class ArchUC:
    def execute(self, request: ArchRequest) -> ArchResult:
        residuals = get_residuals(
            y_true=request.data.ts,
            y_pred=get_full_predict(request.data.ts, request.data.forecasts)
        )
        lmval, lmpval, fval, fpval = acorr_lm(residuals ** 2, ddof=2)
        return ArchResult(
            lmval=lmval,
            lmpval=lmpval,
            fval=fval,
            fpval=fpval,
        )