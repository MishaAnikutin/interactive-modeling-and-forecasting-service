from src.core.domain.stat_test.interface import StatTestService, PValue, TestStatistic
from src.core.domain.stat_test.supported_stat_tests import SupportedStationaryTests


class StationaryTestsFactory:
    _registry: dict[str, type[StatTestService]] = {}

    @classmethod
    def register(cls, test_name: SupportedStationaryTests):
        def wrapper(stat_test_class: type[StationaryTestsFactory]):
            cls._registry[test_name] = stat_test_class
            return stat_test_class

        return wrapper

    @classmethod
    def calculate(cls, test: SupportedStationaryTests, **kwargs) -> tuple[TestStatistic, PValue]:
        return cls._registry[test]().calculate(**kwargs)
