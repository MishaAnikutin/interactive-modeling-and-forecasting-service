from src.core.application.model_diagnosis.schemas.dm_test import DmTestRequest
from src.core.domain.stat_test.dm_test.result import DmTestResult
from src.infrastructure.adapters.forecast_accuracy_comparison.dm_test import DmTestAdapter
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class DmTestUC:
    def __init__(self, ts_adapter: PandasTimeseriesAdapter, dm_test_adapter: DmTestAdapter):
        self._ts_adapter = ts_adapter
        self._dm_test_adapter = dm_test_adapter

    def execute(self, request: DmTestRequest) -> DmTestResult:
        actual = self._ts_adapter.to_series(request.actual)
        forecast1 = self._ts_adapter.to_series(request.forecast1)
        forecast2 = self._ts_adapter.to_series(request.forecast2)

        result: DmTestResult = self._dm_test_adapter.calculate(
            forecast1=forecast1,
            forecast2=forecast2,
            actual=actual,
            significance_level=request.significance_level,
            h=request.h,
            one_sided=request.one_sided,
            harvey_correction=request.harvey_correction,
            variance_estimator=request.variance_estimator,
        )

        return result

