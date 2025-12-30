from src.core.application.predict_series.schemas.schemas import PredictRequest
from src.core.domain import ForecastResult_V2
from src.infrastructure.adapters.predicting.neural_predict.neural_predict import NeuralPredictAdapter
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter


class BaseNeuralPredict_V2:
    def __init__(
        self,
        ts_aligner: TimeseriesAlignment,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter
        self._ts_aligner = ts_aligner
        self._predict_adapter: NeuralPredictAdapter = None

    def execute(self, model_bytes: bytes, request: PredictRequest) -> ForecastResult_V2:
        target, exog_df = self._ts_aligner.align(request.model_data)
        freq = request.model_data.dependent_variables.data_frequency

        return self._predict_adapter.execute(
            model_weight=model_bytes,
            target=target,
            exog_df=exog_df,
            fit_params=request.fit_params,
            data_frequency=freq
        )
