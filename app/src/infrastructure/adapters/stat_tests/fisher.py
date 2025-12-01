from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import f

from src.core.domain import DataFrequency
from src.core.domain.stat_test import Frequency2SeriesSize, Conclusion, SignificanceLevel
from src.core.domain.stat_test.fisher import FisherTestResult
from src.core.domain.stat_test.fisher.errors import InsufficientDataError, InvalidDateError


class FisherTestAdapter:
    def perform_fisher_test(
            self,
            timeseries: pd.Series,
            frequency: DataFrequency,
            date_boundary: datetime,
            alpha: SignificanceLevel = 0.05
    ) -> List[FisherTestResult]:

        series_size = Frequency2SeriesSize.get(frequency)

        boundary_idx = self._find_boundary_index(timeseries, date_boundary)

        return self._perform_test_series(timeseries, boundary_idx, series_size, alpha)

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
            series_size: int,
            alpha: float
    ) -> List[FisherTestResult]:
        points_after_boundary = len(series) - boundary_idx
        is_short_series = points_after_boundary <= series_size + 1

        if is_short_series:
            return self._perform_short_series_tests(series, boundary_idx, series_size, alpha)
        else:
            return self._perform_long_series_tests(series, boundary_idx, series_size, alpha)

    def _perform_short_series_tests(
            self,
            series: pd.Series,
            boundary_idx: int,
            series_size: int,
            alpha: float
    ) -> List[FisherTestResult]:
        results = []
        series_values = series.astype(float).values

        results.append(self._create_nan_result())

        points_after_boundary = len(series) - boundary_idx
        for i in range(2, points_after_boundary - 1):
            data1 = series_values[-i:]
            data2 = series_values[-boundary_idx:-i]

            if len(data1) < 2 or len(data2) < 2:
                continue

            F_stat = np.var(data1, ddof=1) / np.var(data2, ddof=1)
            df1 = len(data1) - 1
            df2 = len(data2) - 1
            p_val = f.cdf(F_stat, df1, df2)

            results.append(self._create_test_result(p_val, F_stat, alpha))

        results.append(self._create_nan_result())

        while len(results) < series_size:
            results.append(self._create_nan_result())

        return results

    def _perform_long_series_tests(
            self,
            series: pd.Series,
            boundary_idx: int,
            series_size: int,
            alpha: float
    ) -> List[FisherTestResult]:
        results = []
        series_values = series.astype(float).values

        results.append(self._create_nan_result())

        for i in range(2, series_size + 1):
            data1 = series_values[-i:]
            data2 = series_values[-boundary_idx:-i]

            if len(data1) < 2 or len(data2) < 2:
                continue

            F_stat = np.var(data1, ddof=1) / np.var(data2, ddof=1)
            df1 = len(data1) - 1
            df2 = len(data2) - 1
            p_val = f.cdf(F_stat, df1, df2)

            results.append(self._create_test_result(p_val, F_stat, alpha))

        if len(results) < series_size:
            results.append(self._create_nan_result())

        return results

    def _create_nan_result(self) -> FisherTestResult:
        return FisherTestResult(
            p_value=float('nan'),
            statistic=float('nan'),
            conclusion=Conclusion.fail_to_reject
        )

    def _create_test_result(self, p_value: float, statistic: float, alpha: float) -> FisherTestResult:
        conclusion = (
            Conclusion.reject
            if p_value < alpha
            else Conclusion.fail_to_reject
        )

        return FisherTestResult(
            p_value=p_value,
            statistic=statistic,
            conclusion=conclusion
        )
