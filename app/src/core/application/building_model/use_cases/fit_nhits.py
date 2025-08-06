import pandas as pd

from src.core.application.building_model.schemas.nhits import NhitsParams, NhitsFitResult, NhitsFitRequest
from src.infrastructure.adapters.model_storage import IModelStorage
from src.infrastructure.adapters.modeling.nhits import NhitsAdapter
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter


class FitNhitsUC:
    def __init__(
        self,
        storage: IModelStorage,
        model_adapter: NhitsAdapter,
        ts_aligner: TimeseriesAlignment,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter
        self._ts_aligner = ts_aligner
        self._model_adapter = model_adapter
        self._storage = storage

    def execute(self, request: NhitsFitRequest) -> NhitsFitResult:
        target, exog_df = self._ts_aligner.align(request.model_data)

        model_result: NhitsFitResult = self._model_adapter.fit(
            target=target,
            exog=exog_df,
            nhits_params=request.hyperparameters,
            fit_params=request.fit_params,
            data_frequency=request.model_data.dependent_variables.data_frequency
        )

        model_id, model_path = self._storage.save(model_result)

        return NhitsFitResult(
            forecasts=model_result.forecasts,
            model_metrics=model_result.model_metrics,
            weight_path=model_path,
            model_id=model_id,
        )