from src.core.application.model_diagnosis.schemas.wilcoxon import WilcoxonRequest
from src.core.domain.stat_test.wilcoxon.result import WilcoxonResult
from src.infrastructure.adapters.equality_of_distribution.wilcoxon import WilcoxonAdapter
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class WilcoxonUC:
    def __init__(self, ts_adapter: PandasTimeseriesAdapter, wilcoxon_adapter: WilcoxonAdapter):
        self._ts_adapter = ts_adapter
        self._wilcoxon_adapter = wilcoxon_adapter

    def execute(self, request: WilcoxonRequest) -> WilcoxonResult:
        forecast = self._ts_adapter.to_series(request.forecast)
        actual = self._ts_adapter.to_series(request.actual)

        result: WilcoxonResult = self._wilcoxon_adapter.calculate(
            forecast=forecast,
            actual=actual,
            significance_level=request.significance_level,
            alternative=request.alternative,
            nan_policy=request.nan_policy,
            correction = request.correction,
            method = request.method,
            zero_method = request.zero_method,
        )

        return result
