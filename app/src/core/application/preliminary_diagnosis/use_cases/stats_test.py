import pandas as pd

from src.core.application.preliminary_diagnosis.schemas.stats_test import StatTestRequest, StatTestResult
from src.core.domain import Timeseries
from src.infrastructure.adapters.stationarity_testing.factory import StationarityTestsFactory


class StationarityUC:
    def __init__(self, stationarity_tests_factory: StationarityTestsFactory):
        self.stationarity_tests_factory = stationarity_tests_factory

    @staticmethod
    def form_series(ts: Timeseries) -> pd.Series:
        return pd.Series(
            data=ts.values,
            index=ts.dates
        )

    def execute(self, request: StatTestRequest) -> list[StatTestResult]:
        test_metrics = self.stationarity_tests_factory.apply(
            stat_tests=request.test_names,
            ts=self.form_series(request.ts),
            alpha=request.alpha,
        )
        return test_metrics