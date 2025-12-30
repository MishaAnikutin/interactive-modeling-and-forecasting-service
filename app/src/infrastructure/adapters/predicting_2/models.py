from .neural_predict import NeuralPredictAdapter_V2


class PredictGruAdapter_V2(NeuralPredictAdapter_V2):
    model_name = "GRU"

class PredictLstmAdapter_V2(NeuralPredictAdapter_V2):
    model_name = "LSTM"

class PredictNhitsAdapter_V2(NeuralPredictAdapter_V2):
    model_name = "NHITS"
