from src.core.application.preliminary_diagnosis.schemas.stats_test import StatTestRequest, StatTestResult
from src.infrastructure.adapters.stationarity_testing.factory import StationarityTestsFactory


class StationarityUC:
    def __init__(self, stationarity_tests_factory: StationarityTestsFactory):
        self.stationarity_tests_factory = stationarity_tests_factory

    def execute(self, request: StatTestRequest) -> StatTestResult:
        raise NotImplementedError