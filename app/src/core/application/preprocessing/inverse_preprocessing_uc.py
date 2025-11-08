from src.core.domain import Timeseries
from src.core.application.preprocessing.preprocess_scheme import InversePreprocessingRequest

from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
from src.infrastructure.factories.preprocessing import PreprocessFactory


class InversePreprocessUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
        preprocess_factory: PreprocessFactory,
    ):
        self._ts_adapter = ts_adapter
        self._preprocess_factory = preprocess_factory

    def execute(self, request: InversePreprocessingRequest) -> Timeseries:
        x = self._ts_adapter.to_series(request.ts)

        for transformation in reversed(request.transformations):
            x = self._preprocess_factory.inverse(x, transformation)

        ts = self._ts_adapter.from_series(x, request.ts.data_frequency)

        return ts
