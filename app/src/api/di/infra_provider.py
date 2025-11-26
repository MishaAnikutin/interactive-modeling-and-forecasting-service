from dishka import Provider, Scope, provide

from src.infrastructure.adapters.distributions import HistogramEstimator, DensityEstimator, EmpiricalDistribution
from src.infrastructure.adapters.preliminary_diagnosis.fao import FaoAdapter
from src.infrastructure.adapters.preliminary_diagnosis.kim_andrews import KimAndrewsAdapter
from src.infrastructure.factories.distributions import DistributionFactory
from src.infrastructure.factories.metrics import MetricsFactory

from src.infrastructure.adapters.modeling import (
    ArimaxAdapter,
    GruAdapter,
    LstmAdapter,
    NhitsAdapter
)

from src.infrastructure.adapters.predicting.arimax import PredictArimaxAdapter
from src.infrastructure.adapters.predicting.neural_predict.models import PredictGruAdapter, PredictNhitsAdapter, PredictLstmAdapter
from src.infrastructure.factories.statistics import StatisticsFactory
from src.infrastructure.factories.preprocessing import PreprocessFactory
from src.infrastructure.adapters.serializer import PickleSerializer, ModelSerializer
from src.infrastructure.adapters.archiver import ModelArchiver, ZipArchiver
from src.infrastructure.factories.validation import ValidationVisitor
from src.infrastructure.adapters.timeseries import (
    PandasTimeseriesAdapter,
    TimeseriesAlignment,
    FrequencyDeterminer,
    TimeseriesTrainTestSplit,
    TimeseriesExtender
)
from src.infrastructure.adapters.dist_fit.dist_fit import DistFit
from src.infrastructure.interactors.correlation import CorrelationInteractor

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

    statistics_fabric = provide(StatisticsFactory, provides=StatisticsFactory)

    dist_fit = provide(DistFit, provides=DistFit)

    histogram_estimator = provide(HistogramEstimator, provides=HistogramEstimator)
    density_estimator = provide(DensityEstimator, provides=DensityEstimator)
    distribution_factory = provide(DistributionFactory, provides=DistributionFactory)

    empirical_dist = provide(EmpiricalDistribution, provides=EmpiricalDistribution)

    validation_factory = provide(ValidationVisitor, provides=ValidationVisitor)

    correlation_interactor = provide(CorrelationInteractor, provides=CorrelationInteractor)

    kim_andrews_adapter = provide(KimAndrewsAdapter, provides=KimAndrewsAdapter)

    fao_adapter = provide(FaoAdapter, provides=FaoAdapter)
