from src.core.application.preprocessing.preprocess_scheme import PreprocessingRequest, DiffTransformation
from src.core.domain import Timeseries
from src.infrastructure.adapters.preprocessing.methods import *
from src.infrastructure.adapters.preprocessing.preprocess_factory import PreprocessFactory
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class PreprocessUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
        preprocess_factory: PreprocessFactory,
    ):
        self._ts_adapter = ts_adapter
        self._preprocess_factory = preprocess_factory

    def execute(self, request: PreprocessingRequest) -> Timeseries:
        x = self._ts_adapter.to_series(request.ts)

        for transformation in request.transformations:
            x = self._preprocess_factory.apply(x, transformation)

        ts = self._ts_adapter.from_series(x)

        return ts


if __name__ == "__main__":
    uc = PreprocessUC(
        PandasTimeseriesAdapter(),
        PreprocessFactory()
    )

    result = uc.execute(request=PreprocessingRequest(ts=Timeseries(), transformations=[DiffTransformation(diff_order=1)]))
    print(result)