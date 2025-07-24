from arch.unitroot import ZivotAndrews
from arch.utility.exceptions import InfeasibleTestException
from fastapi import HTTPException

from src.core.application.preliminary_diagnosis.errors.zivot_andrews import SingularMatrix, InvalidMaxLagError, \
    LowCountObservationsError2, LowCountObservationsError, SingularMatrix2
from src.core.application.preliminary_diagnosis.schemas.common import CriticalValues, StatTestResult
from src.core.application.preliminary_diagnosis.schemas.zivot_andrews import ZivotAndrewsParams
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class ZivotAndrewsUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: ZivotAndrewsParams) -> StatTestResult:
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
            if "observations are needed to run an ADF" in str(exc):
                raise HTTPException(
                    status_code=400,
                    detail=LowCountObservationsError().detail
                )
            elif "The maximum lag you are considering" in str(exc):
                raise HTTPException(
                    status_code=400,
                    detail=SingularMatrix().detail
                )
            elif "The number of observations is too small to use the Zivot-Andrews" in str(exc):
                raise HTTPException(
                    status_code=400,
                    detail=LowCountObservationsError2().detail
                )
            elif "The regressor matrix is singular." in str(exc):
                raise HTTPException(
                    status_code=400,
                    detail=SingularMatrix2().detail
                )
            raise exc
        except ValueError as exc:
            if "maxlag should be < nobs" in str(exc):
                raise HTTPException(
                    status_code=400,
                    detail=InvalidMaxLagError().detail
                )
            raise exc
