from src.core.application.predict_series.schemas.schemas import PredictResponse, PredictRequest
from src.infrastructure.adapters.predicting.arimax import PredictArimaxAdapter
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter


class PredictArimaxUC:
    def __init__(
        self,
        ts_aligner: TimeseriesAlignment,
        ts_adapter: PandasTimeseriesAdapter,
        predict_adapter: PredictArimaxAdapter
    ):
        self._ts_adapter = ts_adapter
        self._ts_aligner = ts_aligner
        self._predict_adapter = predict_adapter

    def execute(self, model_bytes: bytes, request: PredictRequest) -> PredictResponse:
        target, exog_df = self._ts_aligner.align(request.predict_params)
        freq = request.predict_params.dependent_variables.data_frequency

        in_sample, out_of_sample = self._predict_adapter.execute(
            model_weight=model_bytes,
            steps=request.forecast_steps,
            target=target,
            exog_df=exog_df,
            data_frequency=freq
        )

        in_sample_predict = self._ts_adapter.from_series(in_sample, freq)
        out_of_sample_predict = self._ts_adapter.from_series(out_of_sample, freq)

        return PredictResponse(
            in_sample_predict=in_sample_predict,
            out_of_sample_predict=out_of_sample_predict
        )

