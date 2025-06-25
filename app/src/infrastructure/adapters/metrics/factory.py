from src.core.domain import Timeseries, Metric
from src.core.domain.metrics.metrics_service import MetricServiceI


class MetricsFactory:
    registry = {}

    @classmethod
    def register(cls):
        def wrapper(metrics_class: MetricServiceI):
            cls.registry[metrics_class.__class__.__name__] = metrics_class
            return metrics_class

        return wrapper

    @classmethod
    def create(
        cls, metric_type: str
    ) -> MetricServiceI:
        return cls.registry[metric_type]()

    @classmethod
    def apply(
        cls, metrics: list[str], **kwargs
    ) -> list[Metric]:
        return [
            cls.registry[metric_type]().apply(kwargs)
            for metric_type in metrics
        ]
