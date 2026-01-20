from src.core.application.preliminary_diagnosis.schemas.statistics import StatisticResult, \
    StatisticsRequest, StatisticsResponse, RusStatMetricEnum
from src.core.domain import Timeseries
from src.infrastructure.adapters.preliminary_diagnosis.statistics import StatisticsAdapter

class StatisticsUC:
    def __init__(self, statistics_adapter: StatisticsAdapter):
        self._statistics_adapter = statistics_adapter

    def execute(self, request: Timeseries, statistic: RusStatMetricEnum) -> StatisticResult:
        return self._statistics_adapter.execute(request, statistic)

    def execute_many(self, request: StatisticsRequest) -> StatisticsResponse:
        return self._statistics_adapter.execute_many(request)

