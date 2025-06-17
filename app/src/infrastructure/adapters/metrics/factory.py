from src.shared.schemas import Timeseries, Metric
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
        cls, metric_type: str, y_pred: Timeseries, y_true: Timeseries
    ) -> MetricServiceI:
        return cls.registry[metric_type](y_pred=y_pred, y_true=y_true)

    @classmethod
    def apply(
        cls, metrics: list[str], y_pred: Timeseries, y_true: Timeseries
    ) -> list[Metric]:
        return [
            cls.registry[metric_type](y_pred=y_pred, y_true=y_true).apply()
            for metric_type in metrics
        ]
