from abc import ABC
from typing import Optional, Callable

from src.core.application.preliminary_diagnosis.schemas.stats_test import StatTestResult
from src.core.domain.statistical_tests.metric import TestMetric


class StationarityTestingService(ABC):
    strategy: Optional[Callable[..., TestMetric]] = None

    def apply(self, **kwargs) -> StatTestResult:
        if self.strategy is not None:
            stat_metric = type(self).strategy(**kwargs)
            return StatTestResult(
                test_name=kwargs.get("test_name"),
                alpha=kwargs.get("alpha"),
                stat_value=stat_metric.stat_value,
                p_value=stat_metric.p_value,
            )
        raise NotImplementedError