from src.core.application.building_model.schemas.autoarima import AutoArimaRequest, AutoArimaResult
from src.core.domain.parameter_selection.gridsearch_result.arimax import ArimaxGridsearchResult
from src.core.domain.stat_test import SignificanceLevel
from src.core.domain.stat_test.supported_stat_tests import SupportedStationaryTests
from src.infrastructure.adapters.archiver import ModelArchiver
from src.infrastructure.adapters.model_parameters_selection.arima_gridsearch import ArimaGridsearch
from src.infrastructure.adapters.model_parameters_selection.parallel_arima_gridsearch import ParallelArimaGridsearch
from src.infrastructure.adapters.modeling import ArimaxAdapter
from src.infrastructure.adapters.serializer import ModelSerializer
from src.infrastructure.adapters.timeseries import TimeseriesAlignment
from src.infrastructure.factories.stationarity.factory import StationaryTestsFactory


# TODO: можно обобщить до GridAutoML(Generic[RequestT, SearchStrategyT, ModelAdapterT])
#  тут бы был
#  AutoArima(GridAutoML[FitArimaxRequest, ArimaGridsearch, ArimaxAdapter]):
#      ...
#  или
#  AutoArima(GridAutoML):
#     RequestT = FitArimaxRequest
#     SearchStrategyT = ArimaGridsearch
#     ModelAdapterT = ArimaxAdapter
#  что по моему банально нагляднее
#  Есть особенность, что для DL понадобится еще значения скоринга на кросс-валидации в зависимости от числа эпох
#  и для бустингов тоже самое но от числа эстиматоров, так что видимо будут разные стратегии AutoML, которые будут
#  отличаться типами возвращаемого результата
class AutoArimaUC:
    def __init__(
            self,
            gridsearch:    ParallelArimaGridsearch,
            ts_aligner:    TimeseriesAlignment,
            archiver:      ModelArchiver,
            serializer:    ModelSerializer,
            arima_adapter: ArimaxAdapter,
            stationary_factory: StationaryTestsFactory,
    ):
        self._ts_aligner = ts_aligner
        self._gridsearch = gridsearch
        self._archiver = archiver
        self._serializer = serializer
        self._arima_adapter = arima_adapter
        self._stationary_factory = stationary_factory

    def execute(self, request: AutoArimaRequest) -> bytes:
        target, exog = self._ts_aligner.align(request.model_data)

        # TODO: перебрать d до стационарности
        d = 0
        while self._is_not_stationary(target, request.stationary_test):
            d += 1

        # TODO: оценивать параметры надо только на обучающей и валидационной выборке НА КРОСС ВАЛИДАЦИИ
        gridsearch_result: ArimaxGridsearchResult = self._gridsearch.fit(
            endog=target,
            exog=exog,
            max_p=request.max_p,
            d=d,
            max_q=request.max_q,
            max_P=request.max_P,
            max_D=request.max_D,
            max_Q=request.max_Q,
            m=request.m,
            scoring=request.scoring
        )

        # TODO: Обучать нужно на всем наборе данных
        arima_fit_result, model_weight = self._arima_adapter.fit(
            target=target,
            exog=exog,
            hyperparameters=gridsearch_result.optimal_params,
            data_frequency=request.model_data.dependent_variables.data_frequency,
            fit_params=request.fit_params
        )

        result = AutoArimaResult(
            arimax_fit_result=arima_fit_result,
            arimax_gridsearch_result=gridsearch_result
        )

        data_dict: dict = result.model_dump()
        model_bytes: bytes = self._serializer.serialize(model_weight)

        archive: bytes = self._archiver.execute(data_dict=data_dict, model_bytes=model_bytes)

        return archive

    def _is_not_stationary(
            self,
            series,
            stat_test: SupportedStationaryTests,
            significance_level: SignificanceLevel = 0.05
    ) -> bool:
        _, p_value = self._stationary_factory.calculate(test=stat_test, series=series)

        return p_value > significance_level
