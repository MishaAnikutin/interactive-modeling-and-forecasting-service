import pandas as pd

from src.core.application.preliminary_diagnosis.schemas.stats_test import StatTestResult
from src.core.domain.statistical_tests.stationarity_testing_service import StationarityTestingService


class StationarityTestsFactory:
    registry: dict[str, type[StationarityTestingService]] = {}

    @classmethod
    def register(cls):
        def wrapper(stat_test_class: type[StationarityTestingService]):
            cls.registry[stat_test_class.__name__] = stat_test_class
            return stat_test_class

        return wrapper

    @classmethod
    def apply(cls, stat_tests: list[str], ts: pd.Series, alpha: float, **kwargs) -> list[StatTestResult]:
        return [
            cls.registry[stat_test_type](ts, alpha).apply(**kwargs)
            for stat_test_type in stat_tests
        ]
