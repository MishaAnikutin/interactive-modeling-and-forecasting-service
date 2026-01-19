from src.core.application.model_diagnosis.schemas.wilcoxon import WilcoxonRequest, WilcoxonResponse
from src.core.domain import Timeseries
from src.core.domain.stat_test.wilcoxon.result import WilcoxonResult
from src.infrastructure.adapters.equality_of_distribution.wilcoxon import WilcoxonAdapter
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
from src.infrastructure.adapters.timeseries.forecast_aligner import ForecastTargetAligner


class WilcoxonUC:
    def __init__(
            self,
            ts_adapter: PandasTimeseriesAdapter,
            wilcoxon_adapter: WilcoxonAdapter,
            target_aligner: ForecastTargetAligner
    ):
        self._ts_adapter       = ts_adapter
        self._wilcoxon_adapter = wilcoxon_adapter
        self._target_aligner   = target_aligner

    def execute(self, request: WilcoxonRequest) -> WilcoxonResponse:
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

        return WilcoxonResponse(
            train_result=train_result,
            validation_result=validation_result,
            test_result=test_result
        )

    def _calculate_test(self, forecast: Timeseries, actual: Timeseries, request: WilcoxonRequest) -> WilcoxonResult:
        return self._wilcoxon_adapter.calculate(
            forecast=self._ts_adapter.to_series(forecast),
            actual=self._ts_adapter.to_series(actual),
            significance_level=request.significance_level,
            alternative=request.alternative,
            nan_policy=request.nan_policy,
            correction=request.correction,
            method=request.method,
            zero_method=request.zero_method,
        )
