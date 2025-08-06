from src.core.application.predict_series.schemas.schemas import PredictGruRequest, PredictResponse
from src.infrastructure.adapters.predicting.gru import PredictGruAdapter
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter


class PredictGruUC:
    def __init__(
        self,
        ts_aligner: TimeseriesAlignment,
        ts_adapter: PandasTimeseriesAdapter,
        predict_adapter: PredictGruAdapter,
    ):
        self._ts_adapter = ts_adapter
        self._ts_aligner = ts_aligner
        self._predict_adapter = predict_adapter

    def execute(self, request: PredictGruRequest) -> PredictResponse:
        target, exog_df = self._ts_aligner.align(request.model_data)

        in_sample, out_of_sample = self._predict_adapter.execute(
            model_weight=request.model_weight,
            params=request.gru_params,
            target=target,
            exog_df=exog_df
        )

        freq = request.model_data.dependent_variables.data_frequency
        in_sample_predict = self._ts_adapter.from_series(in_sample, freq)
        out_of_sample_predict = self._ts_adapter.from_series(out_of_sample, freq)

        return PredictResponse(
            in_sample_predict=in_sample_predict,
            out_of_sample_predict=out_of_sample_predict
        )

