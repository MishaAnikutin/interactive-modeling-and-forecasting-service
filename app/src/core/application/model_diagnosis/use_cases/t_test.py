from src.core.domain.stat_test.ttest.result import TtestResult
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
from src.core.application.model_diagnosis.schemas.ttest import TtestRequest
from src.infrastructure.adapters.equality_of_distribution.ttest import TtestAdapter


class TtestUC:
    def __init__(self, ts_adapter: PandasTimeseriesAdapter, ttest_adapter: TtestAdapter):
        self._ts_adapter = ts_adapter
        self._ttest_adapter = ttest_adapter

    def execute(self, request: TtestRequest) -> TtestResult:
        forecast = self._ts_adapter.to_series(request.forecast)
        actual = self._ts_adapter.to_series(request.actual)

        result: TtestResult = self._ttest_adapter.calculate(
            forecast=forecast,
            actual=actual,
            significance_level=request.significance_level,
            alternative=request.alternative,
            nan_policy=request.nan_policy,
        )

        return result
