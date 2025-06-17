from dishka import Provider, Scope, provide

from src.infrastructure import (
    PandasTimeseriesAdapter,
    TimeseriesAlignment,
    FrequencyDeterminer,
    TimeseriesTrainTestSplit,
    ArimaxAdapter,
    IModelStorage,
    MockModelStorage,
    MetricsFactory,
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
