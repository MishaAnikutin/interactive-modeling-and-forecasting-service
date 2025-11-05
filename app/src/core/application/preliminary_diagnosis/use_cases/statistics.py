from src.core.application.preliminary_diagnosis.schemas.statistics import StatisticResult, StatisticsEnum
from src.core.domain import Timeseries
from src.infrastructure.factories.statistics.factory import StatisticsFactory
import numpy as np

class StatisticsUC:
    def __init__(self, statistics_fabric: StatisticsFactory):
        self._statistics_fabric = statistics_fabric

    def execute(self, request: Timeseries, statistic: StatisticsEnum) -> StatisticResult:
        ts = np.array(request.values)
        ts = ts[~np.isnan(ts)]
        value = self._statistics_fabric.get_value(ts=ts, statistic=statistic)
        return StatisticResult(value=round(value, 4))
