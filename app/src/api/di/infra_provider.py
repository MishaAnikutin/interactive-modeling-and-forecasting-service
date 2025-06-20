from dishka import Provider, Scope, provide

from src.infrastructure.adapters.metrics import MetricsFactory
from src.infrastructure.adapters.model_storage import IModelStorage, MockModelStorage
from src.infrastructure.adapters.modeling import ArimaxAdapter
from src.infrastructure.adapters.timeseries import (
    PandasTimeseriesAdapter,
    TimeseriesAlignment,
    FrequencyDeterminer,
    TimeseriesTrainTestSplit,
)


class InfraProvider(Provider):
    scope = Scope.REQUEST

    pandas_ts_adapter = provide(
        PandasTimeseriesAdapter, provides=PandasTimeseriesAdapter
    )
    ts_alignment = provide(TimeseriesAlignment, provides=TimeseriesAlignment)
    freq_determiner = provide(FrequencyDeterminer, provides=FrequencyDeterminer)
    ts_spliter = provide(TimeseriesTrainTestSplit, provides=TimeseriesTrainTestSplit)

    arimax = provide(ArimaxAdapter, provides=ArimaxAdapter)

    model_storage = provide(MockModelStorage, provides=IModelStorage)

    metrics_factory = provide(MetricsFactory, provides=MetricsFactory)
