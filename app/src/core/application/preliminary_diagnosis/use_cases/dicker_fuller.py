from statsmodels.tsa.stattools import adfuller

from src.core.application.preliminary_diagnosis.schemas.dickey_fuller import DickeyFullerParams, DickeyFullerResult
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class DickeuFullerUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: DickeyFullerParams) -> DickeyFullerResult:
        ts = self._ts_adapter.to_series(ts_obj=request.ts)
        if not request.autolag:
            icbest = None
            adfstat, pvalue, usedlag, nobs, critvalues = adfuller(
                x=ts.values,
                regression=request.regression,
                maxlag=request.max_lags,
                autolag=request.autolag,
            )
        else:
            adfstat, pvalue, usedlag, nobs, critvalues, icbest = adfuller(
                x=ts.values,
                regression=request.regression,
                maxlag=request.max_lags,
            )
        return DickeyFullerResult(
            stat_value=adfstat,
            p_value=pvalue,
            usedlag=usedlag,
            nobs=nobs,
            critical_values=critvalues,
            information_criterion_max_value=icbest
        )
