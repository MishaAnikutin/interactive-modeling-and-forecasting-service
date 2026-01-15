from datetime import datetime
from typing import List, Optional

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

        return self._perform_test_series(timeseries, boundary_idx, series_size, float(alpha))

    def _find_boundary_index(self, series: pd.Series, date_boundary: datetime) -> int:
        date_boundary = pd.to_datetime(date_boundary)

        if date_boundary in series.index:
            return series.index.get_loc(date_boundary)

        # ближайшая предыдущая дата
        mask = series.index <= date_boundary
        if not mask.any():
            # если нет ни одной даты ≤ границы — это явная ошибка входа
            raise InvalidDateError(
                f"Boundary date {date_boundary} is earlier than all dates in the series."
            )

        nearest_date = series.index[mask].max()
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
        results: List[FisherTestResult] = []
        series_values = series.astype(float).values

        # первый элемент — всегда NaN без даты
        results.append(self._create_nan_result())

        points_after_boundary = len(series) - boundary_idx
        # счётчик реальных тестов (для вычисления дат, как в коде коллеги)
        test_index = 0

        for i in range(2, points_after_boundary - 1):
            data1 = series_values[-i:]
            data2 = series_values[-boundary_idx:-i]

            if len(data1) < 2 or len(data2) < 2:
                continue

            F_stat = np.var(data1, ddof=1) / np.var(data2, ddof=1)
            df1 = len(data1) - 1
            df2 = len(data2) - 1
            p_val = f.cdf(F_stat, df1, df2)

            # дата по логике коллеги: df.iloc[-3 - i, 0]
            dt = self._get_result_datetime(series.index, test_index)
            results.append(self._create_test_result(p_val, F_stat, alpha, dt))

            test_index += 1

        # последний элемент — NaN без даты (как "хвост" в вашем текущем коде)
        results.append(self._create_nan_result())

        # добиваем список NaN'ами до series_size
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
        results: List[FisherTestResult] = []
        series_values = series.astype(float).values

        # первый элемент — всегда NaN без даты
        results.append(self._create_nan_result())

        # здесь количество тестов = series_size - 1 (как a - 1 в коде коллеги)
        test_index = 0

        for i in range(2, series_size + 1):
            data1 = series_values[-i:]
            data2 = series_values[-boundary_idx:-i]

            if len(data1) < 2 or len(data2) < 2:
                continue

            F_stat = np.var(data1, ddof=1) / np.var(data2, ddof=1)
            df1 = len(data1) - 1
            df2 = len(data2) - 1
            p_val = f.cdf(F_stat, df1, df2)

            # дата по логике коллеги: df.iloc[-3 - i, 0]
            dt = self._get_result_datetime(series.index, test_index)
            results.append(self._create_test_result(p_val, F_stat, alpha, dt))

            test_index += 1

        # если по какой-то причине тестов оказалось меньше series_size,
        # добавляем один NaN результат без даты (как и раньше)
        if len(results) < series_size:
            results.append(self._create_nan_result())

        return results

    def _get_result_datetime(
            self,
            index: pd.DatetimeIndex,
            test_index: int
    ) -> Optional[datetime]:
        """
        Восстановление даты теста по логике коллеги:

        date = df.iloc[-3 - i, 0]  где i = 0, 1, 2, ...

        Здесь:
        - index — DatetimeIndex временного ряда,
        - test_index — порядковый номер теста (0, 1, 2, ...).

        Используется index[-3 - test_index].
        """
        pos = -3 - test_index

        # Защита от некорректных индексов (на всякий случай, хотя при корректном a не понадобится)
        if pos < -len(index):
            return None

        ts = index[pos]
        # Pandas Timestamp нормально сериализуется как datetime,
        # но если нужно строго datetime.datetime:
        if hasattr(ts, "to_pydatetime"):
            return ts.to_pydatetime()
        return ts  # Timestamp тоже наследуется от datetime

    def _create_nan_result(self) -> FisherTestResult:
        return FisherTestResult(
            datetime=None,
            p_value=float('nan'),
            statistic=float('nan'),
            conclusion=Conclusion.fail_to_reject
        )

    def _create_test_result(
            self,
            p_value: float,
            statistic: float,
            alpha: float,
            dt: Optional[datetime]
    ) -> FisherTestResult:
        conclusion = (
            Conclusion.reject
            if p_value < alpha
            else Conclusion.fail_to_reject
        )

        return FisherTestResult(
            datetime=dt,
            p_value=p_value,
            statistic=statistic,
            conclusion=conclusion
        )
