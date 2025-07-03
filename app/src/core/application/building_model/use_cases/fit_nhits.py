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
        target = pd.Series(
            index=request.dependent_variables.dates,
            data=request.dependent_variables.values,
            name=request.dependent_variables.name
        )
        exog_df = (
            None if request.explanatory_variables is None
            else self._ts_aligner.compare(timeseries_list=request.explanatory_variables)
        )

        model_result: NhitsFitResult = self._model_adapter.fit(
            target=target,
            exog=None,
            nhits_params=request.hyperparameters,
            fit_params=request.fit_params,
        )

        model_id, model_path = self._storage.save(model_result)

        return NhitsFitResult(
            forecasts=model_result.forecasts,
            model_metrics=model_result.model_metrics,
            weight_path=model_path,
            model_id=model_id,
        )