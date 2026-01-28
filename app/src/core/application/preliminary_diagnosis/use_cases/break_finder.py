from src.core.application.preliminary_diagnosis.schemas.break_finder import BreakFinderRequest, BreakFinderResponse
from src.infrastructure.adapters.structural_shifts.break_finder import BreakFinderAdapter
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class BreakFinderUC:
    def __init__(self, break_finder_adapter: BreakFinderAdapter, timeseries_adapter: PandasTimeseriesAdapter):
        self._break_finder_adapter = break_finder_adapter
        self._timeseries_adapter = timeseries_adapter

    def execute(self, request: BreakFinderRequest) -> BreakFinderResponse:
        endog = self._timeseries_adapter.to_series(request.timeseries)

        breakpoints = self._break_finder_adapter.fit(
            endog=endog,
            trim=request.trim,
            gap=request.gap,
            n_breaks=request.n_breaks,
            criterion=request.criterion.value,
            intercept=request.intercept,
            break_intercept=request.break_intercept,
            trend=request.trend,
            break_trend=request.break_trend,
            seasons=request.seasons
        )

        return breakpoints
