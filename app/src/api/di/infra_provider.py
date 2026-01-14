from dishka import Provider, Scope, provide

from src.infrastructure.adapters.distributions import HistogramEstimator, DensityEstimator, EmpiricalDistribution
from src.infrastructure.adapters.modeling_2.nhits import NhitsAdapter_V2
from src.infrastructure.adapters.modeling_2.lstm import LstmAdapter_V2
from src.infrastructure.adapters.modeling_2.gru import GruAdapter_V2
from src.infrastructure.adapters.predicting_2.models import PredictNhitsAdapter_V2, PredictLstmAdapter_V2, \
    PredictGruAdapter_V2
from src.infrastructure.adapters.preliminary_diagnosis.fao import FaoAdapter
from src.infrastructure.adapters.preliminary_diagnosis.kim_andrews import KimAndrewsAdapter
from src.infrastructure.adapters.stat_tests.fisher import FisherTestAdapter
from src.infrastructure.adapters.stat_tests.student import StudentTestAdapter
from src.infrastructure.adapters.stat_tests.two_sigma import TwoSigmaTestAdapter
from src.infrastructure.adapters.timeseries.split_windows import WindowSplitter
from src.infrastructure.adapters.timeseries.windows_creation import WindowsCreation
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
    nhits2 = provide(NhitsAdapter_V2, provides=NhitsAdapter_V2)
    nhits_predictor = provide(PredictNhitsAdapter, provides=PredictNhitsAdapter)
    nhits_predictor_2 = provide(PredictNhitsAdapter_V2, provides=PredictNhitsAdapter_V2)

    lstm = provide(LstmAdapter, provides=LstmAdapter)
    lstm2 = provide(LstmAdapter_V2, provides=LstmAdapter_V2)
    lstm_predictor = provide(PredictLstmAdapter, provides=PredictLstmAdapter)
    lstm_predictor_2 = provide(PredictLstmAdapter_V2, provides=PredictLstmAdapter_V2)

    gru = provide(GruAdapter, provides=GruAdapter)
    gru2 = provide(GruAdapter_V2, provides=GruAdapter_V2)
    gru_predictor = provide(PredictGruAdapter, provides=PredictGruAdapter)
    gru_predictor_2 = provide(PredictGruAdapter_V2, provides=PredictGruAdapter_V2)

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

    student_adapter = provide(StudentTestAdapter, provides=StudentTestAdapter)
    fisher_adapter = provide(FisherTestAdapter, provides=FisherTestAdapter)
    two_sigma_adapter = provide(TwoSigmaTestAdapter, provides=TwoSigmaTestAdapter)

    windows_creation = provide(WindowsCreation, provides=WindowsCreation)
    windows_splitter = provide(WindowSplitter, provides=WindowSplitter)
