from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from src.core.domain import (
    Timeseries,
    FitParams,
    Forecasts,
    ModelMetrics,
)


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
    n_stacks: int = Field(default=3, description="Число стеков", ge=1)
    n_blocks: list[int] = Field(default=[1, 1, 1],)
    n_pool_kernel_size: list[int] = Field(default=[2, 2, 1],)
    pooling_mode: PoolingMode = Field(default=PoolingMode.MaxPool1d,)
    interpolation_mode: InterpMode = Field(default=InterpMode.Linear,)
    loss: LossEnum = Field(default=LossEnum.MAE,)
    valid_loss: LossEnum = Field(default=LossEnum.MAE,)
    activation: ActivationType = Field(default=ActivationType.ReLU)
    max_steps: int = Field(default=100, ge=1, description="Максимум итераций обучения", le=5000)
    early_stop_patience_steps: int = Field(default=-1, ge=-1, description="Patience для early-stopping", le=5000)
    val_check_steps: int = Field(default=50, ge=0, description="Проверка валидации каждые n шагов", le=5000)
    learning_rate: float = Field(default=1e-3, gt=0, description="Шаг обучения")
    scaler_type: ScalerType = Field(default=ScalerType.Identity, description="Тип скейлера")

    @model_validator(mode='after')
    def validate_list_lengths(self):
        if (
            len(self.n_blocks) != self.n_stacks or
            len(self.n_pool_kernel_size) != self.n_stacks or
            len(self.n_blocks) != len(self.n_pool_kernel_size)
        ):
            raise ValueError(
                "All lists must have same length: "
                f"stack_types={self.n_stacks}, "
                f"n_blocks={len(self.n_blocks)}, "
                f"n_pool_kernel_size={len(self.n_pool_kernel_size)}")
        return self

    @model_validator(mode='after')
    def validate_pool_kernel_sizes(self):
        if any(k < 1 for k in self.n_pool_kernel_size):
            raise ValueError("All n_pool_kernel_size values must be >= 1")
        return self

    @model_validator(mode='after')
    def validate_loss(self):
        losses = ["MAE", "MSE", "RMSE", "MAPE"]
        if self.loss not in losses:
            raise ValueError(f"Loss '{self.loss}' is not supported. Supported losses are: {losses}")
        elif self.valid_loss not in losses:
            raise ValueError(f"Loss '{self.valid_loss}' is not supported. Supported losses are: {losses}")
        return self


class NhitsFitRequest(BaseModel):
    dependent_variables: Timeseries
    explanatory_variables: Optional[List[Timeseries]]
    hyperparameters: NhitsParams
    fit_params: FitParams


class NhitsFitResult(BaseModel):
    forecasts: Forecasts          # прогнозы
    model_metrics: ModelMetrics   # рассчитанные метрики
    weight_path: str              # путь к сохранённым весам
    model_id: str                 # идентификатор модели