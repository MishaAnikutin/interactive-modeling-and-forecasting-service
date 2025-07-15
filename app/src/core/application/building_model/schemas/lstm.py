from typing import Optional, List

from pydantic import BaseModel, Field

from src.core.application.building_model.schemas.nhits import ScalerType, LossEnum
from src.core.domain import Timeseries, FitParams, Forecasts, ModelMetrics


class LstmParams(BaseModel):
    inference_input_size: Optional[int]= Field(default=None, description='Maximum sequence length for truncated inference. Default None uses input_size history.')
    h_train: int = Field(default=1, description='Maximum sequence length for truncated train backpropagation. Default 1.')
    encoder_n_layers: int = Field(default=2, description='Number of layers for the LSTM.')
    encoder_hidden_size: int = Field(default=200, description="units for the LSTM's hidden state size.")
    encoder_dropout: float = Field(default=0.0, description='dropout regularization applied to LSTM outputs.')
    decoder_hidden_size: int = Field(default=200, description="size of hidden layer for the MLP decoder.size of hidden layer for the MLP decoder.")
    decoder_layers: int = Field(default=2, description='number of layers for the MLP decoder.')
    recurrent: bool = Field(default=False, description='whether to exclude the target variable from the input.')
    loss: str = Field(default=LossEnum.MAE,)
    valid_loss: str = Field(default=LossEnum.MAE,)
    max_steps: int = Field(default=100, ge=1, description="Максимум итераций обучения", le=5000)
    learning_rate: float = Field(default=1e-3, gt=0, description="Шаг обучения")
    early_stop_patience_steps: int = Field(default=-1, ge=-1, description="Patience для early-stopping", le=5000)
    val_check_steps: int = Field(default=50, ge=0, description="Проверка валидации каждые n шагов", le=5000)
    scaler_type: ScalerType = Field(default=ScalerType.Robust, description="Тип скейлера")


class LstmFitRequest(BaseModel):
    dependent_variables: Timeseries
    explanatory_variables: Optional[List[Timeseries]]
    hyperparameters: LstmParams
    fit_params: FitParams


class LstmFitResult(BaseModel):
    forecasts: Forecasts          # прогнозы
    model_metrics: ModelMetrics   # рассчитанные метрики
    weight_path: str              # путь к сохранённым весам
    model_id: str                 # идентификатор модели