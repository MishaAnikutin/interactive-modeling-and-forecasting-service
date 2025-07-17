from statsmodels.tsa.seasonal import STL

from src.core.application.generating_series.schemas.stl_decomposition import STLDecompositionRequest, \
    STLDecompositionResult
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class STLDecompositionUC:
    def __init__(
            self,
            ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: STLDecompositionRequest) -> STLDecompositionResult:
        ts = self._ts_adapter.to_series(ts_obj=request.ts)
        result = STL(
            ts,
            period=request.params.period,
            seasonal=request.params.seasonal,
            trend=request.params.trend,
            low_pass=request.params.low_pass,
            seasonal_deg=request.params.seasonal_deg,
            trend_deg=request.params.trend_deg,
            low_pass_deg=request.params.low_pass_deg,
            robust=request.params.robust,
            seasonal_jump=request.params.seasonal_jump,
            trend_jump=request.params.trend_jump,
            low_pass_jump=request.params.low_pass_jump,
        ).fit()

        freq = request.ts.data_frequency

        return STLDecompositionResult(
            observed=self._ts_adapter.from_series(result.observed, freq=freq),
            trend=self._ts_adapter.from_series(result.trend, freq=freq),
            seasonal=self._ts_adapter.from_series(result.seasonal, freq=freq),
            resid=self._ts_adapter.from_series(result.resid, freq=freq),
        )

