import pandas as pd

from src.core.application.preliminary_diagnosis.schemas.statistics import StatisticResult, \
    StatisticsRequest, StatisticsResponse, RusStatMetricEnum, SplitOption, get_russian_metric
from src.core.domain import Timeseries
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
from src.infrastructure.factories.statistics.factory import StatisticsFactory
import numpy as np

class StatisticsAdapter:
    def __init__(
            self,
            statistics_fabric: StatisticsFactory,
            pandas_adapter: PandasTimeseriesAdapter,
    ):
        self._statistics_fabric = statistics_fabric
        self._pandas_adapter = pandas_adapter

    def execute(self, request: Timeseries, statistic: RusStatMetricEnum) -> StatisticResult:
        ts = np.array(request.values)
        ts = ts[~np.isnan(ts)]
        return self._statistics_fabric.get_value(ts=ts, statistic=statistic)

    def execute_many(self, request: StatisticsRequest) -> StatisticsResponse:
        rus_metrics = [get_russian_metric(m) for m in request.metrics]
        ts = self._pandas_adapter.to_dataframe(request.timeseries)

        split_map = {  # Словарь для выбора числа групп по варианту split
            SplitOption.QUARTILE: 4,
            SplitOption.QUINTILE: 5,
            SplitOption.DECILE: 10
        }
        parts = split_map.get(request.split_option, None)
        if parts:
            ts['group'] = pd.qcut(ts[request.timeseries.name], q=parts, labels=False, duplicates='drop')
        else:
            ts['group'] = 'Весь ряд'

        result_dict = {}
        for group_num, group_df in ts.groupby('group'):
            if parts:
                group_name = f"{group_num + 1} {request.split_option.value}"
            else:
                group_name = 'Весь ряд'
            df = group_df[request.timeseries.name].to_numpy()
            values = [
                self._statistics_fabric.get_value(
                    ts=df,
                    statistic=stat
                ) for stat in rus_metrics
            ]
            result_dict[group_name] = dict(zip(rus_metrics, values))
        return StatisticsResponse(results=result_dict)