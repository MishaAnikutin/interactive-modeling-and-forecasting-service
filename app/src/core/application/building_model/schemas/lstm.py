from typing import Optional, List

from pydantic import BaseModel, Field, model_validator
from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MAPE

from src.core.application.building_model.errors.lstm import HiddenSizeError
from src.core.application.building_model.schemas.nhits import ScalerType, LossEnum
from src.core.domain import Timeseries, FitParams, Forecasts, ModelMetrics


class LstmParams(BaseModel):
    input_size: int = Field(
        default=-1,
        ge=-1,
        le=5000,
        title="Размер входного окна обучения",
        description='Размер входного окна обучения. '
                    'input_size + h + размер тестовой выборки должно быть <= размер обучающей выборки',
    )
    inference_input_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=5000,
        title="Размер входного окна инференса",
        description='Размер входного окна инференса'
    )
    h_train: int = Field(
        default=1,
        ge=0,
        le=5000,
        title="Длина обратного распространения ошибки",
        description='Длина обратного распространения ошибки. '
                    'input_size + h_train + размер тестовой выборки должно быть <= размер обучающей выборки'
    )
    encoder_n_layers: int = Field(
        default=2,
        gt=0,
        le=100,
        title="Количество слоёв LSTM",
        description='Количество слоёв LSTM'
    )
    encoder_hidden_size: int = Field(
        default=200,
        gt=1,
        le=5000,
        title="Размер скрытого состояния LSTM",
        description="Размер скрытого состояния LSTM "
                    "Размер скрытого слоя (hidden size) должен быть больше размера проекции (proj size). "
                    "Размер проекции (proj size) равен 1, "
                    "если параметр recurrent установлен в True, и 0 в противном случае. "
                    "Решение: Убедитесь, что размер скрытого слоя больше значения proj size "
                    "(1 при recurrent=True или 0 при recurrent=False)."
    )
    encoder_dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        title="Dropout",
        description='Dropout'
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
        title="Patience для early-stopping. Валидационная выборка должна быть не пустой, "
              "если ранняя остановка включена (early_stop_patience_steps > 0)",
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
            raise ValueError(HiddenSizeError().detail)
        return self

    @model_validator(mode='after')
    def validate_valid_loss(self):
        if self.valid_loss is None:
            self.valid_loss = self.loss
        return self


class LstmFitRequest(BaseModel):
    dependent_variables: Timeseries = Field(
        default=Timeseries(name="Зависимая переменная"),
        title="Зависимая переменная",
        description="Каждый заявленный тип частотности должен соответствовать определенному системой. "
                    "Ряд должен иметь постоянную частотность. "
                    "Разрешенные частотность: [Y, Q, M, D]. "
                    "Каждая дата должна является последним днем месяца, если это не дневные данные. "
                    "Ряд должен быть не пустой."
    )
    explanatory_variables: Optional[List[Timeseries]] = Field(
        default=[Timeseries(name="Объясняющая переменная"), ],
        title="Список объясняющих переменных",
        description="Каждый заявленный тип частотности должен соответствовать определенному системой. "
                    "Тип частотности экзогенной переменной должен быть равен частотности зависимой. "
                    "Ряд должен иметь постоянную частотность. "
                    "Разрешенные частотность: [Y, Q, M, D]. "
                    "Каждая дата должна является последним днем месяца, если это не дневные данные. "
                    "Ряд должен быть не пустой."
    )
    hyperparameters: LstmParams = Field(title="Параметры модели LSTM")
    fit_params: FitParams = Field(
        title="Общие параметры обучения",
        description="Горизонт прогноза + размер тестовой выборки должен быть больше 0. "
                    "Размер валидационной выборки должен быть 0 или "
                    "больше или равен величины горизонт прогнозирования + размер тестовой выборки. "
                    "input_size + h + размер тестовой выборки должно быть <= размер обучающей выборки. "
                    "input_size + h_train + размер тестовой выборки должно быть <= размер обучающей выборки. "
    )


class LstmFitResult(BaseModel):
    forecasts: Forecasts = Field(title="Прогнозы")
    model_metrics: ModelMetrics = Field(title="Метрики модели")
    weight_path: str = Field(default="example.pth", title="Путь до весов модели")
    model_id: str = Field(default="example", title="Идентификатор модели")