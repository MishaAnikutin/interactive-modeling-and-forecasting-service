from src.api.v1.building_model.schemas import ArimaxFitRequest, ArimaxFitResult

from src.infrastructure.adapters import (
    ArimaxAdapter,
    PandasTimeseriesAdapter,
    TimeseriesAlignment,
    IModelStorage,
)


class FitArimaxUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
        model_adapter: ArimaxAdapter,
        storage: IModelStorage,
        ts_aligner: TimeseriesAlignment,
    ):
        self._ts_adapter = ts_adapter
        self._ts_aligner = ts_aligner
        self._model_adapter = model_adapter
        self._storage = storage

    def execute(self, request: ArimaxFitRequest) -> ArimaxFitResult:
        target_df = self._ts_adapter.to_dataframe(request.dependent_variable)
        exog_df = None

        if request.explanatory_variables is not None:
            exog_df = self._ts_aligner.compare(
                timeseries_list=[
                    self._ts_adapter.to_dataframe(ts)
                    for ts in request.explanatory_variables
                ]
            )

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
