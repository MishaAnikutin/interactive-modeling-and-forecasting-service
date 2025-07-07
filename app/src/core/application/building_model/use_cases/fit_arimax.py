import pandas as pd

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
        self._ts_aligner.is_ts_freq_equal_to_expected(request.dependent_variables)
        if request.explanatory_variables:
            df = self._ts_aligner.compare(
                timeseries_list=request.explanatory_variables,
                target=request.dependent_variables
            )

            target = df[request.dependent_variables.name]
            if type(target) == pd.DataFrame:
                target = target.iloc[:, 0]
            exog_df = df.drop(columns=[request.dependent_variables.name])
            if exog_df.empty:
                exog_df = None
        else:
            target = self._ts_adapter.to_series(request.dependent_variables)
            exog_df = None

        model_result: ArimaxFitResult = self._model_adapter.fit(
            target=target,
            exog=exog_df,
            arimax_params=request.hyperparameters,
            fit_params=request.fit_params,
            data_frequency=request.dependent_variables.data_frequency,
        )

        model_id, model_path = self._storage.save(model_result)

        return ArimaxFitResult(
            forecasts=model_result.forecasts,
            coefficients=model_result.coefficients,
            model_metrics=model_result.model_metrics,
            weight_path=model_path,
            model_id=model_id,
        )
