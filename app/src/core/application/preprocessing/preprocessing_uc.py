from src.core.application.preprocessing.preprocess_scheme import (
    PreprocessingRequest,
    PreprocessingResponse,
    PreprocessContext
)
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

    def execute(self, request: PreprocessingRequest) -> PreprocessingResponse:
        x = self._ts_adapter.to_series(request.ts)
        contexts: list[PreprocessContext] = list()

        for step, transformation in enumerate(request.transformations):
            x, context = self._preprocess_factory.apply(x, transformation)

            if context is not None:
                context.step = step + 1
                contexts.append(context)

        # Уберем пропуски с краев
        if (first_valid_index := x.first_valid_index()) is not None:
            x = x.loc[first_valid_index:]

        if (last_valid_index := x.last_valid_index()) is not None:
            x = x.loc[:last_valid_index]

        ts = self._ts_adapter.from_series(x, request.ts.data_frequency)

        return PreprocessingResponse(preprocessed_ts=ts, contexts=contexts)
