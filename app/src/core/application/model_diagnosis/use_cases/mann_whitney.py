from src.core.application.model_diagnosis.schemas.mannwhitney import MannWhitneyRequest
from src.core.domain.stat_test.mann_whitney.result import MannWhitneyResult
from src.infrastructure.adapters.equality_of_distribution.mann_whitney import MannWhitneyAdapter
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class MannWhitneyUC:
    def __init__(self, ts_adapter: PandasTimeseriesAdapter, mann_whitney_adapter: MannWhitneyAdapter):
        self._ts_adapter = ts_adapter
        self._mann_whitney_adapter = mann_whitney_adapter

    def execute(self, request: MannWhitneyRequest) -> MannWhitneyResult:
        forecast = self._ts_adapter.to_series(request.forecast)
        actual = self._ts_adapter.to_series(request.actual)

        result: MannWhitneyResult = self._mann_whitney_adapter.calculate(
            forecast=forecast,
            actual=actual,
            use_continuity=request.use_continuity,
            alternative=request.alternative,
            method=request.method,
            nan_policy=request.nan_policy,
            significance_level=request.significance_level,
        )

        return result
