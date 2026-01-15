import numpy as np
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
        df = self._series_to_dataframe(timeseries)
        boundary_idx = self._find_boundary_index(df, date_boundary)
        series_size = Frequency2SeriesSize.get(frequency)

        return self._perform_test_series(df, boundary_idx, series_size, equal_var, alpha)

    def _series_to_dataframe(self, series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({
            'date': series.index,
            'obs': series.astype(float)
        })

    def _find_boundary_index(self, df: pd.DataFrame, date_boundary: datetime) -> int:
        date_boundary = pd.to_datetime(date_boundary)

        mask = df['date'].dt.date == date_boundary.date()
        if not mask.any():
            dmin, dmax = df['date'].min(), df['date'].max()
            raise InvalidDateError(
                f"index_date не найден. Ищем {date_boundary.date()}. "
                f"Диапазон в данных: {dmin.date()} — {dmax.date()}."
            )
        return np.where(mask)[0][0]  # позиция первого True в маске (int)

    def _perform_test_series(
            self,
            df: pd.DataFrame,
            boundary_idx: int,
            series_size: int,
            equal_var: bool,
            alpha: float
    ) -> List[StudentTestResult]:
        points_after = len(df) - boundary_idx  # исправлено: int - int
        if points_after <= series_size + 1:
            return self._perform_short_series_tests(df, boundary_idx, equal_var, alpha)
        else:
            return self._perform_long_series_tests(df, boundary_idx, series_size, equal_var, alpha)

    def _perform_short_series_tests(
            self,
            df: pd.DataFrame,
            boundary_idx: int,
            equal_var: bool,
            alpha: float
    ) -> List[StudentTestResult]:
        results = []

        # тест для последней точки
        if boundary_idx < len(df) - 1:
            test_result = ttest_1samp(df.obs.iloc[boundary_idx:-1], df.obs.iloc[-1])
            results.append(self._create_test_result(test_result, alpha, df.iloc[-1, 0]))

        # тесты для тела (ttest_ind)
        for i in range(2, len(df) - boundary_idx - 1):
            if boundary_idx < len(df) - i:
                test_result = ttest_ind(df.obs.iloc[boundary_idx:-i], df.obs.iloc[-i:], equal_var=equal_var)
                results.append(self._create_test_result(test_result, alpha, df.iloc[-2 - (len(results)), 0]))

        # тест для первой точки
        if boundary_idx < len(df) - 2:
            test_result = ttest_1samp(df.obs.iloc[boundary_idx + 1:-1], df.obs.iloc[boundary_idx])
            results.append(self._create_test_result(test_result, alpha, df.iloc[boundary_idx, 0]))

        return self._pad_results(results, series_size)

    def _perform_long_series_tests(
            self,
            df: pd.DataFrame,
            boundary_idx: int,
            series_size: int,
            equal_var: bool,
            alpha: float
    ) -> List[StudentTestResult]:
        results = []

        # тест для последней точки
        if boundary_idx < len(df) - 1:
            test_result = ttest_1samp(df.obs.iloc[boundary_idx:-1], df.obs.iloc[-1])
            results.append(self._create_test_result(test_result, alpha, df.iloc[-1, 0]))

        # тесты для тела (ttest_ind)
        for i in range(2, series_size + 1):
            if boundary_idx < len(df) - i:
                test_result = ttest_ind(df.obs.iloc[boundary_idx:-i], df.obs.iloc[-i:], equal_var=equal_var)
                results.append(self._create_test_result(test_result, alpha, df.iloc[-2 - (len(results)), 0]))

        return self._pad_results(results, series_size)

    def _create_test_result(self, test_result, alpha: float, date: datetime) -> StudentTestResult:
        conclusion = Conclusion.reject if test_result.pvalue < alpha else Conclusion.fail_to_reject
        return StudentTestResult(
            date=pd.to_datetime(date),
            p_value=float(test_result.pvalue),
            statistic=float(test_result.statistic),
            conclusion=conclusion
        )

    def _pad_results(self, results: List[StudentTestResult], series_size: int) -> List[StudentTestResult]:
        while len(results) < series_size:
            results.append(StudentTestResult(
                date=pd.NaT,
                p_value=float('nan'),
                statistic=float('nan'),
                conclusion=Conclusion.fail_to_reject
            ))
        return results
