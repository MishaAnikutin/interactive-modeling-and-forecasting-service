from src.core.application.building_model.schemas.nhits import NhitsFitRequest, NhitsFitResponse
from src.infrastructure.adapters.modeling.nhits import NhitsAdapter
from src.infrastructure.adapters.serializer import ModelSerializer
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter


class FitNhitsUC:
    def __init__(
        self,
        model_adapter: NhitsAdapter,
        ts_aligner: TimeseriesAlignment,
        ts_adapter: PandasTimeseriesAdapter,
        model_serializer: ModelSerializer,
    ):
        self._ts_adapter = ts_adapter
        self._ts_aligner = ts_aligner
        self._model_adapter = model_adapter
        self._model_serializer = model_serializer

    def execute(self, request: NhitsFitRequest) -> NhitsFitResponse:
        target, exog_df = self._ts_aligner.align(request.model_data)

        model_result, model_weight = self._model_adapter.fit(
            target=target,
            exog=exog_df,
            nhits_params=request.hyperparameters,
            fit_params=request.fit_params,
            data_frequency=request.model_data.dependent_variables.data_frequency
        )

        serialized_model_weight: str = self._model_serializer.serialize(model_weight)

        return NhitsFitResponse(
            fit_result=model_result,
            serialized_model_weight=serialized_model_weight,
        )