from src.core.application.building_model.schemas.gru import GruFitRequest
from src.infrastructure.adapters.archiver import ModelArchiver
from src.infrastructure.adapters.modeling.gru import GruAdapter
from src.infrastructure.adapters.serializer import ModelSerializer
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter


class FitGruUC:
    def __init__(
        self,
        model_adapter: GruAdapter,
        ts_aligner: TimeseriesAlignment,
        ts_adapter: PandasTimeseriesAdapter,
        archiver: ModelArchiver,
        serializer: ModelSerializer,
    ):
        self._ts_adapter = ts_adapter
        self._ts_aligner = ts_aligner
        self._model_adapter = model_adapter
        self._serializer = serializer
        self._archiver = archiver

    def execute(self, request: GruFitRequest) -> bytes:
        target, exog_df = self._ts_aligner.align(request.model_data)

        model_result, model_weight = self._model_adapter.fit(
            target=target,
            exog=exog_df,
            gru_params=request.hyperparameters,
            fit_params=request.fit_params,
            data_frequency=request.model_data.dependent_variables.data_frequency
        )

        data_dict: dict = model_result.model_dump()
        model_bytes: bytes = self._serializer.serialize(model_weight)

        archive: bytes = self._archiver.execute(data_dict=data_dict, model_bytes=model_bytes)

        return archive