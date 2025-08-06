import pandas as pd

from src.core.application.building_model.schemas.lstm import LstmFitResult, LstmFitRequest
from src.infrastructure.adapters.model_storage import IModelStorage
from src.infrastructure.adapters.modeling.lstm import LstmAdapter
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter


class FitLstmUC:
    def __init__(
        self,
        storage: IModelStorage,
        model_adapter: LstmAdapter,
        ts_aligner: TimeseriesAlignment,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter
        self._ts_aligner = ts_aligner
        self._model_adapter = model_adapter
        self._storage = storage

    def execute(self, request: LstmFitRequest) -> LstmFitResult:
        target, exog_df = self._ts_aligner.align(request.model_data)

        model_result: LstmFitResult = self._model_adapter.fit(
            target=target,
            exog=exog_df,
            lstm_params=request.hyperparameters,
            fit_params=request.fit_params,
            data_frequency=request.model_data.dependent_variables.data_frequency
        )

        model_id, model_path = self._storage.save(model_result)

        return LstmFitResult(
            forecasts=model_result.forecasts,
            model_metrics=model_result.model_metrics,
            weight_path=model_path,
            model_id=model_id,
        )