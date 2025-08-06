import base64

import pandas as pd

from src.core.application.building_model.schemas.arimax import ArimaxFitRequest, ArimaxFitResult, ArimaxFitResponse

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
        model_serializer: ModelSerializer,  # FIXME: по идее сам сериализатор должен быть скрыт в
                                            #  инфраструктурный слой. Т.к. тут неправильно будет написать
                                            #  даже ModelSerializer[statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper]
    ):
        self._ts_adapter = ts_adapter
        self._ts_aligner = ts_aligner
        self._model_adapter = model_adapter
        self._model_serializer = model_serializer

    def execute(self, request: ArimaxFitRequest) -> ArimaxFitResult:
        target, exog_df = self._ts_aligner.align(request.model_data)

        # FIXME: тут по идее инфраструктурный слой протекает в бизнес логику.
        #  Если так возвращать model_weight то бизнес логика зависит от statsmodels
        #  так что до четверга делаем так, потом зарефачим
        model_result, model_weight = self._model_adapter.fit(
            target=target,
            exog=exog_df,
            arimax_params=request.hyperparameters,
            fit_params=request.fit_params,
            data_frequency=request.dependent_variables.data_frequency,
        )

        serialized_model_weight: str = self._model_serializer.serialize(model_weight)

        return ArimaxFitResponse(
            fit_result=model_result,
            serialized_model_weight=serialized_model_weight
        )
