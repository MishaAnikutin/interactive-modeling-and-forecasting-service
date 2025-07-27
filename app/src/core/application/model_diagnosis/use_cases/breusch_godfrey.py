import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.tsatools import lagmat

from src.core.application.model_diagnosis.schemas.breusch_godfrey import BreuschGodfreyRequest
from src.core.application.model_diagnosis.schemas.common import DiagnosticsResult
from src.core.application.model_diagnosis.use_cases.interface import ResidAnalysisInterface
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter
from src.shared.full_predict import get_full_predict
from src.shared.get_residuals import get_residuals


def _acorr_breusch_godfrey(
    residuals: np.ndarray,
    exog: np.ndarray | None = None,
    nlags: int | None = None,
) -> tuple[float, float, float, float]:
    x = np.asarray(residuals).squeeze()
    nobs = x.shape[0]
    if nlags is None:
        nlags = min(10, nobs // 5)

    # дополняем нулями, чтобы сделать лаги
    x_ext = np.concatenate((np.zeros(nlags), x))

    # матрица лагов остатков
    lagged_resid = lagmat(x_ext[:, None], nlags, trim="both")
    nobs = lagged_resid.shape[0]

    # константа + лаги остатков
    aux_exog = np.c_[np.ones((nobs, 1)), lagged_resid]

    # экзогенные переменные модели (если есть) дописываем к aux_exog
    if exog is not None:
        if len(exog) != nobs:
            raise ValueError("Размерность exog не совпадает с количеством наблюдений")
        aux_exog = np.column_stack((exog, aux_exog))

    # укороченный ряд остатков (из-за лагов)
    x_short = x_ext[-nobs:]

    # вспомогательная регрессия
    resols = OLS(x_short, aux_exog).fit()

    # F-статистика
    k_vars = aux_exog.shape[1]
    ft = resols.f_test(np.eye(nlags, k_vars, k_vars - nlags))
    f_val = float(np.squeeze(ft.fvalue))
    f_pval = float(np.squeeze(ft.pvalue))

    # LM-статистика
    lm_val = nobs * resols.rsquared
    lm_pval = stats.chi2.sf(lm_val, nlags)

    return lm_val, lm_pval, f_val, f_pval


class AcorrBreuschGodfreyUC(ResidAnalysisInterface):
    def __init__(
            self,
            ts_aligner: TimeseriesAlignment,
            ts_adapter: PandasTimeseriesAdapter,
    ):
        super().__init__(ts_aligner, ts_adapter)

    def execute(self, request: BreuschGodfreyRequest) -> DiagnosticsResult:
        target, exog_df = self._aligned_data(request.data.target, request.data.exog)
        residuals = get_residuals(
            y_true=target,
            y_pred=get_full_predict(target, request.data.forecasts),
        )
        exog_arr = None
        if exog_df is not None and not exog_df.empty:
            exog_arr = exog_df.to_numpy(dtype=float)
        lm, lm_p, f_val, f_p = _acorr_breusch_godfrey(
            residuals=residuals,
            exog=exog_arr,
            nlags=request.nlags,
        )
        return DiagnosticsResult(
            lmval=lm,
            lmpval=lm_p,
            fval=f_val,
            fpval=f_p,
        )
