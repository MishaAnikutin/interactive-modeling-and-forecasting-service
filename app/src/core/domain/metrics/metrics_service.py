from abc import ABC

from .metric import Metric
from src.core.domain.timeseries import Timeseries


class MetricServiceI(ABC):
    strategy = None

    def __init__(self, y_pred: Timeseries, y_true: Timeseries):
        self._y_pred = y_pred
        self._y_true = y_true

    def apply(self, *args, **kwargs) -> Metric:
        if self.strategy is not None:
            return Metric(
                type=self.__class__.__name__,
                value=self.strategy(self._y_pred, self._y_true),
            )
        raise NotImplementedError
