from src.core.domain import Metric
from src.core.domain.metrics.metrics_service import MetricServiceI


class MetricsFactory:
    registry: dict[str, type[MetricServiceI]] = {}

    @classmethod
    def register(cls):
        def wrapper(metrics_class: type[MetricServiceI]):
            cls.registry[metrics_class.__name__] = metrics_class
            return metrics_class

        return wrapper

    @classmethod
    def create(cls, metric_type: str) -> MetricServiceI:
        return cls.registry[metric_type]()

    @classmethod
    def apply(cls, metrics: list[str], **kwargs) -> list[Metric]:
        return [
            cls.registry[metric_type]().apply(**kwargs)
            for metric_type in metrics
        ]
