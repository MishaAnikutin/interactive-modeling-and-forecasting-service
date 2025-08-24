from src.core.application.predict_series.use_cases.neural_predict import BaseNeuralPredict
from src.infrastructure.adapters.predicting.neural_predict.models import PredictNhitsAdapter
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter


class PredictNhitsUC(BaseNeuralPredict):
    def __init__(
        self,
        ts_aligner: TimeseriesAlignment,
        ts_adapter: PandasTimeseriesAdapter,
        predict_adapter: PredictNhitsAdapter,
    ):
        super().__init__(ts_adapter=ts_adapter, ts_aligner=ts_aligner)
        self._predict_adapter = predict_adapter

