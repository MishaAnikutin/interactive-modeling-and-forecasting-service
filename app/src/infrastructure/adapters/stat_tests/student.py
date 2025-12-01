import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp
from typing import List
from datetime import datetime

from src.core.domain import DataFrequency
from src.core.domain.stat_test import Frequency2SeriesSize
from src.core.domain.stat_test.conclusion import Conclusion
from src.core.domain.stat_test.student import InvalidDateError, StudentTestResult


class StudentTestAdapter:
    def perform_student_test(
            self,
            timeseries: pd.Series,
            frequency: DataFrequency,
            date_boundary: datetime,
            equal_var: bool,
            alpha: float
    ) -> List[StudentTestResult]:
        series_size = Frequency2SeriesSize.get(frequency)
        boundary_idx = self._find_boundary_index(timeseries, date_boundary)

        return self._perform_test_series(timeseries, boundary_idx, series_size, equal_var, alpha)

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
            equal_var: bool,
            alpha: float
    ) -> List[StudentTestResult]:
        results = []
        series = series.astype(float)
        is_short_series = len(series) - boundary_idx <= series_size + 1

        if is_short_series:
            results.extend(self._perform_short_series_tests(series, boundary_idx, equal_var, alpha))
        else:
            results.extend(self._perform_long_series_tests(series, boundary_idx, series_size, equal_var, alpha))

        while len(results) < series_size:
            results.append(StudentTestResult(
                p_value=float('nan'),
                statistic=float('nan'),
                conclusion=Conclusion.fail_to_reject
            ))

        return results

    def _perform_short_series_tests(
            self,
            series: pd.Series,
            boundary_idx: int,
            equal_var: bool,
            alpha: float
    ) -> List[StudentTestResult]:
        results = []

        if boundary_idx < len(series) - 1:
            data_before = series.iloc[boundary_idx:-1]  # от boundary_idx до предпоследнего
            last_value = series.iloc[-1]
            last_point_test = ttest_1samp(data_before, last_value)
            results.append(self._create_test_result(last_point_test, alpha))

        available_tests = len(series) - boundary_idx - 2
        for i in range(2, min(available_tests + 1, len(series) - boundary_idx - 1)):
            data1 = series.iloc[boundary_idx:-i]  # от boundary_idx до -i
            data2 = series.iloc[-i:]  # последние i значений
            test_result = ttest_ind(data1, data2, equal_var=equal_var)
            results.append(self._create_test_result(test_result, alpha))

        if boundary_idx < len(series) - 2:
            data_after = series.iloc[boundary_idx + 1:-1]  # от boundary_idx+1 до предпоследнего
            boundary_value = series.iloc[boundary_idx]
            first_point_test = ttest_1samp(data_after, boundary_value)
            results.append(self._create_test_result(first_point_test, alpha))

        return results

    def _perform_long_series_tests(
            self,
            series: pd.Series,
            boundary_idx: int,
            series_size: int,
            equal_var: bool,
            alpha: float
    ) -> List[StudentTestResult]:
        results = []

        if boundary_idx < len(series) - 1:
            data_before = series.iloc[boundary_idx:-1]
            last_value = series.iloc[-1]
            last_point_test = ttest_1samp(data_before, last_value)
            results.append(self._create_test_result(last_point_test, alpha))

        for i in range(2, series_size + 1):
            if boundary_idx < len(series) - i:
                data1 = series.iloc[boundary_idx:-i]
                data2 = series.iloc[-i:]
                test_result = ttest_ind(data1, data2, equal_var=equal_var)
                results.append(self._create_test_result(test_result, alpha))

        return results

    def _create_test_result(self, test_result, alpha: float) -> StudentTestResult:
        conclusion = (
            Conclusion.reject
            if test_result.pvalue < alpha
            else Conclusion.fail_to_reject
        )

        return StudentTestResult(
            p_value=test_result.pvalue,
            statistic=test_result.statistic,
            conclusion=conclusion
        )
