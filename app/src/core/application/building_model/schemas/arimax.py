from typing import List, Optional

from pydantic import BaseModel, Field

from src.core.domain import Timeseries, FitParams, Forecasts, Coefficient, ModelMetrics


class ArimaxParams(BaseModel):
    p: int = Field(default=0, ge=0)
    d: int = Field(default=0, ge=0)
    q: int = Field(default=0, ge=0)


class ArimaxFitRequest(BaseModel):
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
        default=[Timeseries(name="Объясняющая переменная"),],
        title="Список объясняющих переменных",
        description="Каждый заявленный тип частотности должен соответствовать определенному системой. "
                    "Тип частотности экзогенной переменной должен быть равен частотности зависимой. "
                    "Ряд должен иметь постоянную частотность. "
                    "Разрешенные частотность: [Y, Q, M, D]. "
                    "Каждая дата должна является последним днем месяца, если это не дневные данные. "
                    "Ряд должен быть не пустой."
    )
    hyperparameters: ArimaxParams = Field(title='Параметры модели ARIMAX')
    fit_params: FitParams = Field(title="Общие параметры обучения", description="train_boundary должна быть раньше val_boundary ")


class ArimaxFitResult(BaseModel):
    forecasts: Forecasts = Field(title="Прогнозы")
    coefficients: List[Coefficient] = Field(title="Список коэффициентов")
    model_metrics: ModelMetrics = Field(title="Метрики модели")
    weight_path: str = Field(default="example.pth", title="Путь до весов модели")
    model_id: str = Field(default="example", title="Идентификатор модели")
