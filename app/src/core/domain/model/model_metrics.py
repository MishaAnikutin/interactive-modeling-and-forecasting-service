from pydantic import BaseModel

from src.core.domain.metrics.metric import Metric


class ModelMetrics(BaseModel):
    train_metrics: list[Metric]
    test_metrics: list[Metric]
