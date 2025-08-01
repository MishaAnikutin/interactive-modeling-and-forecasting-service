from fastapi import HTTPException
from statsmodels.tsa.stattools import kpss

from src.core.application.preliminary_diagnosis.schemas.common import CriticalValues, StatTestResult
from src.core.application.preliminary_diagnosis.schemas.kpss import KpssParams
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class KpssUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: KpssParams) -> StatTestResult:
        ts = self._ts_adapter.to_series(ts_obj=request.ts)
        try:
            kpss_stat, p_value, nlags, crit_dict = kpss(ts, regression=request.regression, nlags=request.nlags)
            return StatTestResult(
                p_value=p_value,
                stat_value=kpss_stat,
                critical_values=CriticalValues(
                    percent_1=crit_dict['1%'],
                    percent_5=crit_dict['5%'],
                    percent_10=crit_dict['10%'],
                ),
                lags=nlags,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc)
            )