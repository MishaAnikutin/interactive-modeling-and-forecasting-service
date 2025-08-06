from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from src.core.application.building_model.errors.nhits import ListLengthError, KernelSizeError
from src.core.domain import (
    Timeseries,
    FitParams,
    Forecasts,
    ModelMetrics,
)
from src.core.domain.model.model_data import ModelData


class InterpMode(str, Enum):
    Linear = "linear"
    Nearest = "nearest"


class PoolingMode(str, Enum):
    MaxPool1d = "MaxPool1d"
    AvgPool1d = "AvgPool1d"


class ScalerType(str, Enum):
    Robust = "robust"
    Standard = "standard"
    MinMax = "minmax"
    Identity = "identity"
    Revin = "revin"
    Invariant = "invariant"


class LossEnum(str, Enum):
    MAE = "MAE"
    MSE = "MSE"
    RMSE = "RMSE"
    MAPE = "MAPE"


class ActivationType(str, Enum):
    ReLU = "ReLU"
    Softplus = "Softplus"
    Tanh = "Tanh"
    SELU = "SELU"
    LeakyReLU = "LeakyReLU"
    PReLU = "PReLU"
    Sigmoid = "Sigmoid"

class NhitsParams(BaseModel):
    n_stacks: int = Field(
        default=3,
        title="Число стеков",
        ge=1,
        description="n_stacks = len(n_blocks) = len(n_pool_kernel_size)"
    )
    n_blocks: list[int] = Field(
        default=[1, 1, 1],
        title="Число блоков в каждом стеке.",
        description="len(n_blocks) = n_stacks = len(n_pool_kernel_size)"
    )
    n_pool_kernel_size: list[int] = Field(
        default=[2, 2, 1],
        title="Параметр задаёт список размеров окон",
        description="Параметр задаёт список размеров окон, для которых вычисляется максимум или среднее значение. "
                    "len(n_pool_kernel_size) = n_stacks = len(n_blocks). "
                    "Все значения в списке n_pool_kernel_size должны быть больше или равны 1"
    )
    pooling_mode: PoolingMode = Field(
        default=PoolingMode.MaxPool1d,
        title="Pooling mode"
    )
    interpolation_mode: InterpMode = Field(
        default=InterpMode.Linear,
        title="Interpolation mode"
    )
    loss: LossEnum = Field(
        default=LossEnum.MAE,
        title="Название функции ошибки, используемой при обучении"
    )
    valid_loss: LossEnum = Field(
        default=LossEnum.MAE,
        title="Название функции ошибки, используемой при валидации"
    )
    activation: ActivationType = Field(
        default=ActivationType.ReLU,
        title="Название функции активации"
    )
    max_steps: int = Field(default=100, ge=1, description="Максимум итераций обучения", le=5000)
    early_stop_patience_steps: int = Field(default=-1, ge=-1, description="Patience для early-stopping", le=5000)
    val_check_steps: int = Field(default=50, ge=0, description="Проверка валидации каждые n шагов", le=5000)
    learning_rate: float = Field(default=1e-3, gt=0, description="Шаг обучения", le=1.0)
    scaler_type: ScalerType = Field(default=ScalerType.Identity, description="Тип скейлера")

    @model_validator(mode='after')
    def validate_list_lengths(self):
        if (
            len(self.n_blocks) != self.n_stacks or
            len(self.n_pool_kernel_size) != self.n_stacks or
            len(self.n_blocks) != len(self.n_pool_kernel_size)
        ):
            raise ValueError(ListLengthError().detail)
        return self

    @model_validator(mode='after')
    def validate_pool_kernel_sizes(self):
        if any(k < 1 for k in self.n_pool_kernel_size):
            raise ValueError(KernelSizeError().detail)
        return self



class NhitsFitRequest(BaseModel):
    model_data: ModelData = Field(default=ModelData())
    hyperparameters: NhitsParams = Field(title="Параметры модели NHiTS")
    fit_params: FitParams = Field(
        title="Общие параметры обучения",
        description="train_boundary должна быть раньше val_boundary "
                    "Горизонт прогноза + размер тестовой выборки должен быть больше 0. "
                    "Размер валидационной выборки должен быть 0 или "
                    "больше или равен величины горизонт прогнозирования + размер тестовой выборки. "
                    "4 * (h + размер тестовой выборки) должно быть <= размер обучающей выборки. "
    )


class NhitsFitResult(BaseModel):
    forecasts: Forecasts = Field(title="Прогнозы")
    model_metrics: ModelMetrics = Field(title="Метрики модели")
    weight_path: str = Field(default="example.pth", title="Путь до весов модели")
    model_id: str = Field(default="example", title="Идентификатор модели")