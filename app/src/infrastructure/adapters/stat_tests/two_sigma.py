from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from src.core.domain import DataFrequency
from src.core.domain.stat_test import Frequency2SeriesSize
from src.core.domain.stat_test.two_sigma.errors import InsufficientDataError, InvalidDateError
from src.core.domain.stat_test.two_sigma.growth_conclusion import GrowthConclusion
from src.core.domain.stat_test.two_sigma.result import TwoSigmaTestResult


class TwoSigmaTestAdapter:
    def perform_two_sigma_test(
            self,
            timeseries: pd.Series,
            frequency: DataFrequency,
            date_boundary: datetime
    ) -> List[TwoSigmaTestResult]:

        series_size = Frequency2SeriesSize.get(frequency)
        boundary_idx = self._find_boundary_index(timeseries, date_boundary)

        return self._perform_test_series(timeseries, boundary_idx, series_size)

    def _find_boundary_index(self, series: pd.Series, date_boundary: datetime) -> int:
        date_boundary = pd.to_datetime(date_boundary)

        if date_boundary in series.index:
            return series.index.get_loc(date_boundary)

        nearest_date = series.index[series.index <= date_boundary].max()

        return series.index.get_loc(nearest_date)

    def _perform_test_series(
            self,
            series: pd.Series,
            boundary_idx: int,
            series_size: int
    ) -> List[TwoSigmaTestResult]:
        results = []
        series_values = series.astype(float).values

        for j in range(1, series_size + 1):
            growth_rates = []
            for i in range(boundary_idx - series_size - j, boundary_idx - j):
                if i >= series_size:
                    growth_rate = series_values[i] / series_values[i - series_size]
                    growth_rates.append(growth_rate)

            if len(growth_rates) < 2:
                results.append(TwoSigmaTestResult(
                    std=float('nan'),
                    confidence_interval=(float('nan'), float('nan')),
                    conclusion=GrowthConclusion.normal
                ))
                continue

            growth_series = pd.Series(growth_rates)
            diff_series = growth_series.diff(1).dropna()

            if len(diff_series) == 0:
                results.append(TwoSigmaTestResult(
                    std=float('nan'),
                    confidence_interval=(float('nan'), float('nan')),
                    conclusion=GrowthConclusion.normal
                ))
                continue

            current_std = np.std(diff_series)
            mean_growth = np.mean(diff_series)
            ci_lower = mean_growth - 2 * current_std
            ci_upper = mean_growth + 2 * current_std

            is_anomalous = current_std < ci_lower or current_std > ci_upper

            results.append(self._create_test_result(current_std, (ci_lower, ci_upper), is_anomalous))

        return results

    def _create_test_result(self, std: float, confidence_interval: tuple, is_anomalous: bool) -> TwoSigmaTestResult:
        conclusion = (
            GrowthConclusion.anomalous
            if is_anomalous
            else GrowthConclusion.normal
        )

        return TwoSigmaTestResult(
            std=std,
            confidence_interval=confidence_interval,
            conclusion=conclusion
        )
