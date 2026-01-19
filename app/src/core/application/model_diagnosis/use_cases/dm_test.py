from src.core.application.model_diagnosis.schemas.dm_test import DmTestRequest, DmTestResponse
from src.core.domain import Timeseries
from src.core.domain.stat_test.dm_test.result import DmTestResult
from src.infrastructure.adapters.forecast_accuracy_comparison.dm_test import DmTestAdapter
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
from src.infrastructure.adapters.timeseries.forecast_aligner import ForecastTargetAligner


class DmTestUC:
    def __init__(self, ts_adapter: PandasTimeseriesAdapter, target_aligner: ForecastTargetAligner, dm_test_adapter: DmTestAdapter):
        self._ts_adapter = ts_adapter
        self._target_aligner = target_aligner
        self._dm_test_adapter = dm_test_adapter

    def execute(self, request: DmTestRequest) -> DmTestResponse:
        train_target, validation_target, test_target = self._target_aligner.align(
            forecasts=request.forecasts1, target=request.target
        )

        train_result = self._calculate_dm_test(
            forecast1=request.forecasts1.train_predict,
            forecast2=request.forecasts2.train_predict,
            actual   =train_target,
            request  =request
        )

        validation_result = None
        if validation_target is not None:
            validation_result = self._calculate_dm_test(
                forecast1=request.forecasts1.validation_predict,
                forecast2=request.forecasts2.validation_predict,
                actual   =validation_target,
                request  =request
            )

        test_result = None
        if test_target is not None:
            test_result = self._calculate_dm_test(
                forecast1=request.forecasts1.test_predict,
                forecast2=request.forecasts2.test_predict,
                actual   =test_target,
                request  =request
            )

        return DmTestResponse(
            train_result=train_result,
            validation_result=validation_result,
            test_result=test_result
        )

    def _calculate_dm_test(
            self,
            forecast1: Timeseries,
            forecast2: Timeseries,
            actual: Timeseries,
            request: DmTestRequest
    ) -> DmTestResult:
        return self._dm_test_adapter.calculate(
            forecast1=self._ts_adapter.to_series(forecast1),
            forecast2=self._ts_adapter.to_series(forecast2),
            actual=self._ts_adapter.to_series(actual),
            significance_level=request.significance_level,
            h=request.h,
            one_sided=request.one_sided,
            harvey_correction=request.harvey_correction,
            variance_estimator=request.variance_estimator,
        )
