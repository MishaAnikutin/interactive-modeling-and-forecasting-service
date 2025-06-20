from src.core.application.building_model.schemas.arimax import ArimaxFitRequest, ArimaxFitResult
from src.infrastructure.adapters.timeseries import (
    PandasTimeseriesAdapter,
    TimeseriesAlignment,
)

from src.infrastructure.adapters.modeling import ArimaxAdapter
from src.infrastructure.adapters.model_storage import IModelStorage


class FitArimaxUC:
    def __init__(
        self,
        storage: IModelStorage,
        model_adapter: ArimaxAdapter,
        ts_aligner: TimeseriesAlignment,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter
        self._ts_aligner = ts_aligner
        self._model_adapter = model_adapter
        self._storage = storage

    def execute(self, request: ArimaxFitRequest) -> ArimaxFitResult:
        target_df = self._ts_adapter.to_dataframe(request.dependent_variables)

        exog_df = (None
                   if request.explanatory_variables is None
                   else self._ts_aligner.compare(timeseries_list=request.explanatory_variables))

        model_result: ArimaxFitResult = self._model_adapter.fit(
            target=target_df,
            exog=exog_df,
            arimax_params=request.hyperparameters,
            fit_params=request.fit_params,
        )

        model_id, model_path = self._storage.save(model_result)

        return ArimaxFitResult(
            forecasts=model_result.forecasts,
            fit_result=model_result.coefficients,
            model_metrics=model_result.model_metrics,
            weight_path=model_path,
            model_id=model_id,
        )
