from src.core.application.preliminary_diagnosis.schemas.statistics import StatisticResult, \
    StatisticsRequest, StatisticsResponse, RusStatMetricEnum
from src.core.domain import Timeseries
from src.infrastructure.factories.statistics.factory import StatisticsFactory
import numpy as np

class StatisticsAdapter:
    def __init__(
            self,
            statistics_fabric: StatisticsFactory
    ):
        self._statistics_fabric = statistics_fabric

    def execute(self, request: Timeseries, statistic: RusStatMetricEnum) -> StatisticResult:
        ts = np.array(request.values)
        ts = ts[~np.isnan(ts)]
        return self._statistics_fabric.get_value(ts=ts, statistic=statistic)

    def execute_many(self, request: StatisticsRequest) -> StatisticsResponse:
        values = [
            self._statistics_fabric.get_value(
                ts=request.timeseries,
                statistic=stat
            ) for stat in request.metrics
        ]
        return StatisticsResponse(
            results=dict(zip(request.metrics, values))
        )