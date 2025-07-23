from typing import Optional

from pydantic import BaseModel, Field

from src.core.domain.metrics.metric import Metric


class ModelMetrics(BaseModel):
    train_metrics: list[Metric] = Field(title="Метрики на обучающей выборке")
    val_metrics: Optional[list[Metric]] = Field(title="Метрики на валидационной выборке")
    test_metrics: Optional[list[Metric]] = Field(title="Метрики на тестовой выборке")
