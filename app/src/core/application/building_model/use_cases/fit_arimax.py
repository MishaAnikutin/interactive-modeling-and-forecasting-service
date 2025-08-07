from src.core.application.building_model.schemas.arimax import ArimaxFitRequest, ArimaxFitResponse
from src.infrastructure.adapters.archiver import ModelArchiver

from src.infrastructure.adapters.modeling import ArimaxAdapter
from src.infrastructure.adapters.serializer import ModelSerializer
from src.infrastructure.adapters.timeseries import (
    PandasTimeseriesAdapter,
    TimeseriesAlignment,
)


# FIXME: все настолько обобщено, что меняется только тип адаптера модели и схемы входных и выходных данных
#  поэтому по идее можно все обобщить до фабрики моделей. Однако тогда пропадет гибкость
class FitArimaxUC:
    def __init__(
        self,
        model_adapter: ArimaxAdapter,
        ts_aligner: TimeseriesAlignment,
        ts_adapter: PandasTimeseriesAdapter,
        archiver: ModelArchiver,
        serializer: ModelSerializer,        # FIXME: по идее сам сериализатор должен быть скрыт в
                                            #  инфраструктурный слой. Т.к. тут неправильно будет написать
                                            #  даже ModelSerializer[statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper]
    ):
        self._ts_adapter = ts_adapter
        self._ts_aligner = ts_aligner
        self._model_adapter = model_adapter
        self._serializer = serializer
        self._archiver = archiver

    def execute(self, request: ArimaxFitRequest) -> bytes:
        target, exog_df = self._ts_aligner.align(request.model_data)

        # FIXME: тут по идее инфраструктурный слой протекает в бизнес логику.
        #  Если так возвращать model_weight то бизнес логика зависит от statsmodels
        #  так что до четверга делаем так, потом зарефачим
        model_result, model_weight = self._model_adapter.fit(
            target=target,
            exog=exog_df,
            arimax_params=request.hyperparameters,
            fit_params=request.fit_params,
            data_frequency=request.model_data.dependent_variables.data_frequency,
        )

        data_dict: dict = model_result.model_dump()
        model_bytes: bytes = self._serializer.serialize(model_weight)

        archive: bytes = self._archiver.execute(data_dict=data_dict, model_bytes=model_bytes)

        return archive
