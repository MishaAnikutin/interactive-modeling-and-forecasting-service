from src.core.application.model_diagnosis.schemas.mannwhitney import MannWhitneyRequest, MannWhitneyResponse
from src.core.domain import Timeseries
from src.core.domain.stat_test.mann_whitney.result import MannWhitneyResult
from src.infrastructure.adapters.equality_of_distribution.mann_whitney import MannWhitneyAdapter
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
from src.infrastructure.adapters.timeseries.forecast_aligner import ForecastTargetAligner


class MannWhitneyUC:
    def __init__(
            self,
            ts_adapter: PandasTimeseriesAdapter,
            mann_whitney_adapter: MannWhitneyAdapter,
            target_aligner: ForecastTargetAligner
    ):
        self._ts_adapter = ts_adapter
        self._target_aligner = target_aligner
        self._mann_whitney_adapter = mann_whitney_adapter

    def execute(self, request: MannWhitneyRequest) -> MannWhitneyResponse:
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

        return MannWhitneyResponse(
            train_result=train_result,
            validation_result=validation_result,
            test_result=test_result
        )

    def _calculate_test(
            self,
            forecast: Timeseries,
            actual: Timeseries,
            request: MannWhitneyRequest
    ) -> MannWhitneyResult:
            return self._mann_whitney_adapter.calculate(
                forecast=self._ts_adapter.to_series(forecast),
                actual=self._ts_adapter.to_series(actual),
                use_continuity=request.use_continuity,
                alternative=request.alternative,
                method=request.method,
                nan_policy=request.nan_policy,
                significance_level=request.significance_level,
            )
