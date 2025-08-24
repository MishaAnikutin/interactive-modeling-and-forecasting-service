from src.core.application.predict_series.schemas.schemas import PredictResponse, PredictRequest
from src.infrastructure.adapters.predicting.neural_predict.neural_predict import NeuralPredictAdapter
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter


class BaseNeuralPredict:
    def __init__(
        self,
        ts_aligner: TimeseriesAlignment,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter
        self._ts_aligner = ts_aligner
        self._predict_adapter: NeuralPredictAdapter = None

    def execute(self, model_bytes: bytes, request: PredictRequest) -> PredictResponse:
        target, exog_df = self._ts_aligner.align(request.predict_params)
        freq = request.predict_params.dependent_variables.data_frequency

        in_sample, out_of_sample = self._predict_adapter.execute(
            model_weight=model_bytes,
            target=target,
            exog_df=exog_df,
            steps=request.forecast_steps,
            data_frequency=freq
        )

        in_sample_predict = self._ts_adapter.from_series(in_sample, freq)
        out_of_sample_predict = self._ts_adapter.from_series(out_of_sample, freq)

        return PredictResponse(
            in_sample_predict=in_sample_predict,
            out_of_sample_predict=out_of_sample_predict
        )

