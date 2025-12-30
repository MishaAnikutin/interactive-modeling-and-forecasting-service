from src.core.application.predict_series.use_cases.neural_predict import BaseNeuralPredict
from src.core.application.predict_series.use_cases.neural_predict_2 import BaseNeuralPredict_V2
from src.infrastructure.adapters.predicting.neural_predict.models import PredictNhitsAdapter
from src.infrastructure.adapters.predicting_2.models import PredictNhitsAdapter_V2
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


class PredictNhitsUC_V2(BaseNeuralPredict_V2):
    def __init__(
        self,
        ts_aligner: TimeseriesAlignment,
        ts_adapter: PandasTimeseriesAdapter,
        predict_adapter: PredictNhitsAdapter_V2,
    ):
        super().__init__(ts_adapter=ts_adapter, ts_aligner=ts_aligner)
        self._predict_adapter = predict_adapter