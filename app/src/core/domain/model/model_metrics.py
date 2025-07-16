from typing import Optional

from pydantic import BaseModel

from src.core.domain.metrics.metric import Metric


class ModelMetrics(BaseModel):
    train_metrics: list[Metric]
    val_metrics: Optional[list[Metric]]
    test_metrics: Optional[list[Metric]]
