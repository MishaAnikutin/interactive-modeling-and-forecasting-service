from src.core.domain import Timeseries
from src.core.domain.stat_test.ttest.result import TtestResult
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
from src.core.application.model_diagnosis.schemas.ttest import TtestRequest, TtestResponse
from src.infrastructure.adapters.equality_of_distribution.ttest import TtestAdapter
from src.infrastructure.adapters.timeseries.forecast_aligner import ForecastTargetAligner


class TtestUC:
    def __init__(
            self,
            ts_adapter: PandasTimeseriesAdapter,
            ttest_adapter: TtestAdapter,
            target_aligner: ForecastTargetAligner
    ):
        self._ts_adapter     = ts_adapter
        self._ttest_adapter  = ttest_adapter
        self._target_aligner = target_aligner

    def execute(self, request: TtestRequest) -> TtestResult:
        train_target, validation_target, test_target = self._target_aligner.align(
            forecasts=request.forecasts, target=request.target
        )

        train_result = self._calculate_test(
            forecast=request.forecasts.train_predict,
            actual=train_target,
            request=request
        )

        validation_result = None
        if validation_target is not None:
            validation_result = self._calculate_test(
                forecast=request.forecasts.validation_predict,
                actual=validation_target,
                request=request
            )

        test_result = None
        if test_target is not None:
            test_result = self._calculate_test(
                forecast=request.forecasts.test_predict,
                actual=test_target,
                request=request
            )

        return TtestResponse(
            train_result=train_result,
            validation_result=validation_result,
            test_result=test_result
        )

    def _calculate_test(self, forecast: Timeseries, actual: Timeseries, request: TtestRequest) -> TtestResult:
        return self._ttest_adapter.calculate(
            forecast=self._ts_adapter.to_series(forecast),
            actual=self._ts_adapter.to_series(actual),
            significance_level=request.significance_level,
            alternative=request.alternative,
            nan_policy=request.nan_policy,
        )
