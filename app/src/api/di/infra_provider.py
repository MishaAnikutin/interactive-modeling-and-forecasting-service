from dishka import Provider, Scope, provide

from src.infrastructure.adapters.metrics import MetricsFactory

from src.infrastructure.adapters.modeling import (
    ArimaxAdapter,
    GruAdapter,
    LstmAdapter,
    NhitsAdapter
)

from src.infrastructure.adapters.predicting.arimax import PredictArimaxAdapter
from src.infrastructure.adapters.predicting.neural_predict.models import PredictGruAdapter, PredictNhitsAdapter, PredictLstmAdapter
from src.infrastructure.adapters.preliminary_diagnosis import PPplotFactory, KdeFactory
from src.infrastructure.adapters.preprocessing.preprocess_factory import PreprocessFactory
from src.infrastructure.adapters.serializer import PickleSerializer, ModelSerializer
from src.infrastructure.adapters.archiver import ModelArchiver, ZipArchiver
from src.infrastructure.adapters.timeseries import (
    PandasTimeseriesAdapter,
    TimeseriesAlignment,
    FrequencyDeterminer,
    TimeseriesTrainTestSplit,
    TimeseriesExtender
)


class InfraProvider(Provider):
    scope = Scope.REQUEST

    pandas_ts_adapter = provide(
        PandasTimeseriesAdapter, provides=PandasTimeseriesAdapter
    )
    ts_alignment = provide(TimeseriesAlignment, provides=TimeseriesAlignment)
    ts_extender = provide(TimeseriesExtender, provides=TimeseriesExtender)
    freq_determiner = provide(FrequencyDeterminer, provides=FrequencyDeterminer)
    ts_spliter = provide(TimeseriesTrainTestSplit, provides=TimeseriesTrainTestSplit)
    model_serializer = provide(PickleSerializer, provides=ModelSerializer)
    model_archiver = provide(ZipArchiver, provides=ModelArchiver)

    arimax = provide(ArimaxAdapter, provides=ArimaxAdapter)
    arimax_predictor = provide(PredictArimaxAdapter, provides=PredictArimaxAdapter)

    nhits = provide(NhitsAdapter, provides=NhitsAdapter)
    nhits_predictor = provide(PredictNhitsAdapter, provides=PredictNhitsAdapter)

    lstm = provide(LstmAdapter, provides=LstmAdapter)
    lstm_predictor = provide(PredictLstmAdapter, provides=PredictLstmAdapter)

    gru = provide(GruAdapter, provides=GruAdapter)
    gru_predictor = provide(PredictGruAdapter, provides=PredictGruAdapter)

    metrics_factory = provide(MetricsFactory, provides=MetricsFactory)

    preprocess_factory = provide(PreprocessFactory, provides=PreprocessFactory)

    pp_plot_factory = provide(PPplotFactory, provides=PPplotFactory)
    kde_factory = provide(KdeFactory, provides=KdeFactory)
