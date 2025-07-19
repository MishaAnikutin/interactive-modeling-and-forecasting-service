from arch.unitroot import PhillipsPerron

from src.core.application.preliminary_diagnosis.schemas.common import CriticalValues, StatTestResult
from src.core.application.preliminary_diagnosis.schemas.phillips_perron import PhillipsPerronParams
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class PhillipsPerronUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: PhillipsPerronParams) -> StatTestResult:
        ts = self._ts_adapter.to_series(ts_obj=request.ts)
        result = PhillipsPerron(ts, trend=request.trend, lags=request.lags, test_type=request.test_type)
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
