from abc import ABC
from typing import Callable, Optional

from .metric import Metric

class MetricServiceI(ABC):
    strategy: Optional[Callable[..., float]] = None

    def apply(self, **kwargs) -> Metric:
        if self.strategy is not None:
            return Metric(
                type=self.__class__.__name__,
                value=type(self).strategy(y_pred=kwargs["y_pred"], y_true=kwargs["y_true"]),
            )
        raise NotImplementedError
