from arch.unitroot import DFGLS
from arch.utility.exceptions import InfeasibleTestException
from fastapi import HTTPException

from src.core.application.preliminary_diagnosis.schemas.common import CriticalValues, StatTestResult
from src.core.application.preliminary_diagnosis.schemas.df_gls import DfGlsParams
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class DfGlsUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: DfGlsParams) -> StatTestResult:
        ts = self._ts_adapter.to_series(ts_obj=request.ts)
        try:
            result = DFGLS(
                y=ts,
                lags=request.lags,
                max_lags=request.max_lags,
                trend=request.trend,
                method=request.method,
            )
            critvalues = result.critical_values
            return StatTestResult(
                p_value=result.pvalue,
                stat_value=result.stat,
                critical_values=CriticalValues(
                    percent_1=critvalues['1%'],
                    percent_5=critvalues['5%'],
                    percent_10=critvalues['10%'],
                ),
                lags=result.lags,
            )
        except InfeasibleTestException as exc:
            if "The maximum lag you are considering" in str(exc):
                raise HTTPException(
                    status_code=400,
                    detail=str(exc)
                )
            raise exc
        except ValueError as exc:
            if "maxlag should be < nobs" in str(exc):
                raise HTTPException(
                    status_code=400,
                    detail=str(exc)
                )
            raise exc