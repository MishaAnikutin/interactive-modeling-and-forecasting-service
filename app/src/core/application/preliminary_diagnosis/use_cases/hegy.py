from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
from src.shared.hegy import seasonalURoot

from src.core.application.preliminary_diagnosis.schemas.hegy import HegyRequest, HegyResponse, HegyStatistic, \
    HegyStatisticType


class HegyUC:
    def __init__(self, timeseries_adapter: PandasTimeseriesAdapter):
        self._timeseries_adapter = timeseries_adapter

    def execute(self, request: HegyRequest) -> HegyResponse:
        y = self._timeseries_adapter.to_series(request.y)

        result = seasonalURoot(
            y=y,
            max_lag=request.max_lag,
            trend=request.trend.value,
            criteria=request.criteria.value,
            S=request.S,
            stats_only=request.stats_only
        )

        statistics: list[HegyStatistic] = list()

        for stat_type, raw in result.iterrows():

            stat_type = HegyStatisticType(stat_type) if not stat_type.endswith('w') else stat_type
            statistics.append(HegyStatistic(
                type=stat_type,
                test_statistic=raw['stat'],
                p_value=raw['pval']
            ))

        return HegyResponse(statistics=statistics)
