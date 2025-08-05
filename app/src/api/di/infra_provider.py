from dishka import Provider, Scope, provide

from src.infrastructure.adapters.metrics import MetricsFactory
from src.infrastructure.adapters.model_storage import MockModelStorage, IModelStorage

from src.infrastructure.adapters.modeling import (
    ArimaxAdapter,
    GruAdapter,
    LstmAdapter,
    NhitsAdapter
)

from src.infrastructure.adapters.predicting.arimax import PredictArimaxAdapter
from src.infrastructure.adapters.preprocessing.preprocess_factory import PreprocessFactory
from src.infrastructure.adapters.serializer import Base64PickleSerializer, ModelSerializer
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
    model_serializer = provide(Base64PickleSerializer, provides=ModelSerializer)

    arimax = provide(ArimaxAdapter, provides=ArimaxAdapter)
    arimax_predictor = provide(PredictArimaxAdapter, provides=PredictArimaxAdapter)

    nhits = provide(NhitsAdapter, provides=NhitsAdapter)
    lstm = provide(LstmAdapter, provides=LstmAdapter)
    gru = provide(GruAdapter, provides=GruAdapter)

    # TODO: депрекейтед
    model_storage = provide(MockModelStorage, provides=IModelStorage)

    metrics_factory = provide(MetricsFactory, provides=MetricsFactory)

    preprocess_factory = provide(PreprocessFactory, provides=PreprocessFactory)
