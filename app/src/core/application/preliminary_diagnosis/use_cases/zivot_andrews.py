from arch.unitroot import ZivotAndrews
from arch.utility.exceptions import InfeasibleTestException
from fastapi import HTTPException
from statsmodels.tsa.stattools import zivot_andrews

from src.core.application.preliminary_diagnosis.schemas.common import CriticalValues
from src.core.application.preliminary_diagnosis.schemas.zivot_andrews import ZivotAndrewsParams, ZivotAndrewsResult
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class ZivotAndrewsUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: ZivotAndrewsParams) -> ZivotAndrewsResult:
        ts = self._ts_adapter.to_series(ts_obj=request.ts)
        try:
            result = ZivotAndrews(
                y=ts,
                trim=request.trim,
                trend=request.regression,
                max_lags=request.max_lags,
                lags=request.lags,
                method=request.autolag,
            )
            critvalues = result.critical_values
            return ZivotAndrewsResult(
                p_value=result.pvalue,
                stat_value=result.stat,
                critical_values=CriticalValues(
                    percent_1=critvalues['1%'],
                    percent_5=critvalues['5%'],
                    percent_10=critvalues['10%'],
                ),
                lags=result.lags,
                nobs=result.nobs
            )
        except InfeasibleTestException as exc:
            if "observations are needed to run an ADF" in str(exc):
                raise HTTPException(
                    status_code=400,
                    detail=str(exc)
                )
            elif "The maximum lag you are considering" in str(exc):
                raise HTTPException(
                    status_code=400,
                    detail=str(exc)
                )
            elif "The number of observations is too small to use the Zivot-Andrews" in str(exc):
                raise HTTPException(
                    status_code=400,
                    detail=str(exc)
                )
            elif "The regressor matrix is singular." in str(exc):
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
