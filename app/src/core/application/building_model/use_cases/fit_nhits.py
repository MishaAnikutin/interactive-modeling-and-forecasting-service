from src.core.application.building_model.schemas.nhits import NhitsParams, NhitsFitResult, NhitsFitRequest
from src.infrastructure.adapters.model_storage import IModelStorage
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter


class FitNhitsUC:
    def __init__(
        self,
        storage: IModelStorage,
        model_adapter: ...,
        ts_aligner: TimeseriesAlignment,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter
        self._ts_aligner = ts_aligner
        self._model_adapter = model_adapter
        self._storage = storage

    def execute(self, request: NhitsFitRequest) -> NhitsFitResult:
        pass