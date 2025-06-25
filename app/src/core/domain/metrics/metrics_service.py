from abc import ABC

from .metric import Metric
from src.core.domain.timeseries import Timeseries


class MetricServiceI(ABC):
    strategy = None

    def apply(self, *args, **kwargs) -> Metric:
        if self.strategy is not None:
            return Metric(
                type=self.__class__.__name__,
                value=self.strategy(kwargs),
            )
        raise NotImplementedError
