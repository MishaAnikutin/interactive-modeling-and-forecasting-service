from datetime import datetime
from typing import List, Optional, Tuple

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

        # ближайшая предыдущая дата
        mask = series.index <= date_boundary
        if not mask.any():
            raise InvalidDateError(
                f"Boundary date {date_boundary} is earlier than all dates in the series."
            )

        nearest_date = series.index[mask].max()
        return series.index.get_loc(nearest_date)

    def _perform_test_series(
            self,
            series: pd.Series,
            boundary_idx: int,
            series_size: int
    ) -> List[TwoSigmaTestResult]:
        results: List[TwoSigmaTestResult] = []
        series_values = series.astype(float).values

        # счётчик реальных тестов (для вычисления дат, как в коде коллеги)
        test_index = 0

        for j in range(1, series_size + 1):
            growth_rates = []
            for i in range(boundary_idx - series_size - j, boundary_idx - j):
                if i >= series_size:
                    growth_rate = series_values[i] / series_values[i - series_size]
                    growth_rates.append(growth_rate)

            if len(growth_rates) < 2:
                # для NaN результата используем None дату
                results.append(self._create_nan_result())
                continue

            growth_series = pd.Series(growth_rates)
            diff_series = growth_series.diff(1).dropna()

            if len(diff_series) == 0:
                results.append(self._create_nan_result())
                continue

            current_std = np.std(diff_series)
            mean_growth = np.mean(diff_series)
            ci_lower = mean_growth - 2 * current_std
            ci_upper = mean_growth + 2 * current_std

            is_anomalous = current_std < ci_lower or current_std > ci_upper

            # дата по логике коллеги: df.iloc[-1 - test_index, 0]
            dt = self._get_result_datetime(series.index, test_index)
            results.append(self._create_test_result(current_std, (ci_lower, ci_upper), is_anomalous, dt))

            test_index += 1  # увеличиваем только для успешных тестов

        # добиваем список NaN'ами до series_size, если нужно
        while len(results) < series_size:
            results.append(self._create_nan_result())

        return results

    def _get_result_datetime(
            self,
            index: pd.DatetimeIndex,
            test_index: int
    ) -> Optional[datetime]:
        """
        Восстановление даты теста по логике коллеги:

        date = [df.iloc[-1 - i, 0] for i in range(len(result_std))]
        где i = 0, 1, 2, ...

        Здесь:
        - index — DatetimeIndex временного ряда,
        - test_index — порядковый номер теста (0, 1, 2, ...).

        Используется index[-1 - test_index].
        """
        pos = -1 - test_index

        # Защита от некорректных индексов
        if pos < -len(index):
            return None

        ts = index[pos]
        # Pandas Timestamp нормально сериализуется как datetime
        if hasattr(ts, "to_pydatetime"):
            return ts.to_pydatetime()
        return ts  # Timestamp тоже наследуется от datetime

    def _create_test_result(
            self,
            std: float,
            confidence_interval: Tuple[float, float],
            is_anomalous: bool,
            dt: Optional[datetime]
    ) -> TwoSigmaTestResult:
        conclusion = (
            GrowthConclusion.anomalous
            if is_anomalous
            else GrowthConclusion.normal
        )

        return TwoSigmaTestResult(
            datetime=dt,
            std=std,
            confidence_interval=confidence_interval,
            conclusion=conclusion
        )

    def _create_nan_result(self) -> TwoSigmaTestResult:
        """Создаёт NaN-результат с None датой"""
        return TwoSigmaTestResult(
            datetime=None,
            std=float('nan'),
            confidence_interval=(float('nan'), float('nan')),
            conclusion=GrowthConclusion.normal
        )
