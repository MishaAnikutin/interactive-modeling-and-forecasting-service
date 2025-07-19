from statsmodels.tsa.seasonal import seasonal_decompose

from src.core.application.generating_series.schemas.naive_decomposition import NaiveDecompositionRequest, \
    NaiveDecompositionResult
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class NaiveDecompositionUC:
    def __init__(
            self,
            ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: NaiveDecompositionRequest) -> NaiveDecompositionResult:
        ts = self._ts_adapter.to_series(ts_obj=request.ts)

        trend = request.params.extrapolate_trend if request.params.extrapolate_trend is not None else "freq"

        result = seasonal_decompose(
            x=ts,
            model=request.params.model,
            filt=request.params.filt,
            period=request.params.period,
            two_sided=request.params.two_sided,
            extrapolate_trend=trend,
        )

        freq = request.ts.data_frequency

        return NaiveDecompositionResult(
            observed=self._ts_adapter.from_series(result.observed, freq=freq),
            trend=self._ts_adapter.from_series(result.trend, freq=freq),
            seasonal=self._ts_adapter.from_series(result.seasonal, freq=freq),
            resid=self._ts_adapter.from_series(result.resid, freq=freq),
        )
