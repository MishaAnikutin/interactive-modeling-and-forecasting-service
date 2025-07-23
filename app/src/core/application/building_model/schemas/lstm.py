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
        title="Размер входного окна обучения",
        description='Maximum sequence length for truncated train',
    )
    inference_input_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=5000,
        title="Размер входного окна инференса",
        description='Maximum sequence length for truncated inference. Default None uses input_size history.'
    )
    h_train: int = Field(
        default=1,
        ge=0,
        le=5000,
        title="Длина обратного распространения ошибки",
        description='Maximum sequence length for truncated train backpropagation. Default 1.'
    )
    encoder_n_layers: int = Field(
        default=2,
        gt=0,
        le=100,
        title="Количество слоёв LSTM",
        description='Number of layers for the LSTM.'
    )
    encoder_hidden_size: int = Field(
        default=200,
        gt=1,
        le=5000,
        title="Размер скрытого состояния LSTM",
        description="Units for the LSTM's hidden state size."
    )
    encoder_dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        title="Dropout",
        description='Dropout regularization applied to LSTM outputs.'
    )
    decoder_hidden_size: int = Field(
        default=128,
        ge=0,
        le=5000,
        title="Размер скрытого слоя декодера",
    )
    decoder_layers: int = Field(
        default=2,
        ge=0,
        le=100,
        title="Количество слоёв декодера",
    )
    recurrent: bool = Field(
        default=False,
        title="Исключить целевую переменную из входа",
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
        le=5000,
        title="Максимальное число шагов обучения",
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        title="Скорость обучения",
        description="Шаг обучения"
    )
    early_stop_patience_steps: int = Field(
        default=-1,
        ge=-1,
        le=5000,
        title="Patience для early-stopping",
    )
    val_check_steps: int = Field(
        default=50,
        ge=0,
        le=5000,
        title="Число шагов между проверками валидации",
    )
    scaler_type: ScalerType = Field(
        default=ScalerType.Robust,
        title="Тип скейлера",
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