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
    def create(cls, stat_test_type: str) -> StationarityTestingService:
        return cls.registry[stat_test_type]()

    @classmethod
    def apply(cls, stat_tests: list[str], **kwargs) -> list[StatTestResult]:
        return [
            cls.registry[metric_type]().apply(**kwargs)
            for metric_type in stat_tests
        ]
