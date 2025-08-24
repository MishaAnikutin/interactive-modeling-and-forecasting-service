from .neural_predict import NeuralPredictAdapter


class PredictGruAdapter(NeuralPredictAdapter):
    model_name = "GRU"

class PredictLstmAdapter(NeuralPredictAdapter):
    model_name = "LSTM"

class PredictNhitsAdapter(NeuralPredictAdapter):
    model_name = "NHITS"
