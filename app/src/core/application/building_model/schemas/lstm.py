from typing import Optional, List

from pydantic import BaseModel, Field, model_validator
from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MAPE

from src.core.application.building_model.schemas.nhits import ScalerType, LossEnum
from src.core.domain import Timeseries, FitParams, Forecasts, ModelMetrics


class LstmParams(BaseModel):
    input_size: int = Field(
        default=-1,
        ge=-1,
        le=5000,
        description='Maximum sequence length for truncated train',
    )
    inference_input_size: Optional[int]= Field(
        default=None,
        ge=1,
        le=5000,
        description='Maximum sequence length for truncated inference. Default None uses input_size history.'
    )
    h_train: int = Field(
        default=1,
        ge=0,
        le=5000,
        description='Maximum sequence length for truncated train backpropagation. Default 1.'
    )
    encoder_n_layers: int = Field(
        default=2,
        gt=0,
        le=100,
        description='Number of layers for the LSTM.'
    )
    encoder_hidden_size: int = Field(
        default=200,
        gt=1,
        le=5000,
        description="Units for the LSTM's hidden state size."
    )
    encoder_dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description='Dropout regularization applied to LSTM outputs.'
    )
    decoder_hidden_size: int = Field(
        default=128,
        ge=0,
        le=5000,
        description="Size of hidden layer for the MLP decoder.size of hidden layer for the MLP decoder."
    )
    decoder_layers: int = Field(
        default=2,
        ge=0,
        le=100,
        description='number of layers for the MLP decoder.'
    )
    recurrent: bool = Field(
        default=False,
        description='whether to exclude the target variable from the input.'
    )
    loss: LossEnum = Field(
        default=LossEnum.MAE,
        title="Название функции ошибки, используемой при обучении"
    )
    valid_loss: Optional[LossEnum] = Field(
        default=None,
        title="Название функции ошибки, используемой при валидации"
    )
    max_steps: int = Field(
        default=100,
        ge=1,
        description="Максимум итераций обучения",
        le=5000
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        description="Шаг обучения"
    )
    early_stop_patience_steps: int = Field(
        default=-1,
        ge=-1,
        description="Patience для early-stopping", le=5000
    )
    val_check_steps: int = Field(
        default=50,
        ge=0,
        description="Проверка валидации каждые n шагов",
        le=5000
    )
    scaler_type: ScalerType = Field(
        default=ScalerType.Robust,
        description="Тип скейлера"
    )

    @model_validator(mode='after')
    def validate_hidden_size(self):
        loss_map = {
            "MAE": MAE,
            "MSE": MSE,
            "RMSE": RMSE,
            "MAPE": MAPE,
        }
        loss = loss_map[self.loss]()
        proj_size = loss.outputsize_multiplier if self.recurrent else 0
        if proj_size >= self.encoder_hidden_size:
            raise ValueError("proj_size has to be smaller than hidden_size")
        return self

    @model_validator(mode='after')
    def validate_valid_loss(self):
        if self.valid_loss is None:
            self.valid_loss = self.loss
        return self


class LstmFitRequest(BaseModel):
    dependent_variables: Timeseries = Field(
        default=Timeseries(name="Зависимая переменная"),
        title="Зависимая переменная"
    )
    explanatory_variables: Optional[List[Timeseries]] = Field(
        default=[Timeseries(name="Объясняющая переменная"), ],
        title="Список объясняющих переменных"
    )
    hyperparameters: LstmParams = Field(title="Параметры модели LSTM")
    fit_params: FitParams = Field(title="Общие параметры обучения")


class LstmFitResult(BaseModel):
    forecasts: Forecasts = Field(title="Прогнозы")
    model_metrics: ModelMetrics = Field(title="Метрики модели")
    weight_path: str = Field(default="example.pth", title="Путь до весов модели")
    model_id: str = Field(default="example", title="Идентификатор модели")