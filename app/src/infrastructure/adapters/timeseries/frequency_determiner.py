from datetime import date, timedelta
from typing import List

from fastapi import HTTPException

from src.core.application.building_model.errors.alignment import NotSupportedFreqError, NotLastDayOfMonthError, \
    EmptyError
from src.core.domain import DataFrequency


class FrequencyDeterminer:
    """
    Класс-утилита, определяющий дискретность временного ряда.
    Алгоритм анализирует минимальный шаг между соседними метками в днях.
    """

    def _first_check_timestamps(self, timestamps: List[date]) -> None:
        if not timestamps:
            raise HTTPException(
                status_code=400,
                detail=EmptyError().detail
            )

        return None

    def _is_last_day_of_month(self, day: date) -> bool:
        """Проверяет, является ли дата последним днем месяца."""
        next_day = day + timedelta(days=1)
        return next_day.month != date.month

    def _validate_dates(self, timestamps: List[date], freq: DataFrequency) -> None:
        """Проверяет формат дат в соответствии с частотностью."""
        if freq != DataFrequency.day:
            for ts in timestamps:
                if not self._is_last_day_of_month(ts):
                    raise HTTPException(
                        status_code=400,
                        detail=NotLastDayOfMonthError().detail
                    )
        return None

    def _check_day_freq(self, deltas: list[int]) -> None:
        for d in deltas:
            # Для ежедневных данных требование слабее из-за финансовых рынков
            # более элегантно это сделать сейчас хз. По идее можно добавить
            # частотность DB - Daily business. Она допускала бы пропуски по выходным и праздникам
            if d > 4:
                raise HTTPException(
                    status_code=400,
                    detail="Ряд не постоянной частотности: ожидаются ежедневные данные (шаг 1 день)"
                )
        return None

    def _check_month_freq(self, deltas: list[int]) -> None:
        for d in deltas:
            if not (28 <= d <= 31):
                raise HTTPException(
                    status_code=400,
                    detail="Ряд не постоянной частотности: ожидаются ежемесячные данные (шаг 28-31 день)"
                )
        return None

    def _check_quarter_freq(self, deltas: list[int]) -> None:
        for d in deltas:
            if not (89 <= d <= 92):
                raise HTTPException(
                    status_code=400,
                    detail="Ряд не постоянной частотности: ожидаются квартальные данные (шаг 89-92 дня)"
                )
        return None

    def _check_year_freq(self, deltas: list[int]) -> None:
        for d in deltas:
            if not (365 <= d <= 366):
                raise HTTPException(
                    status_code=400,
                    detail="Ряд не постоянной частотности: ожидаются годовые данные (шаг 365-366 дней)"
                )
        return None

    def determine(self, timestamps: List[date]) -> DataFrequency:
        self._first_check_timestamps(timestamps)

        sorted_ts = sorted(timestamps)

        # Обработка ряда из одной даты
        if len(sorted_ts) == 1:
            return DataFrequency.day

        # Вычисление разниц между датами в днях
        deltas_days = []
        for i in range(1, len(sorted_ts)):
            delta = sorted_ts[i] - sorted_ts[i - 1]
            deltas_days.append(delta.days)

        # Определение частотности по первой разнице
        first_delta = deltas_days[0]

        if first_delta <= 4:  # Потому что на бирже в выходные нет данных
            freq = DataFrequency.day
            self._check_day_freq(deltas_days)
        elif 28 <= first_delta <= 31:
            freq = DataFrequency.month
            self._check_month_freq(deltas_days)
        elif 89 <= first_delta <= 92:
            freq = DataFrequency.quart
            self._check_quarter_freq(deltas_days)
        elif 365 <= first_delta <= 366:
            self._check_year_freq(deltas_days)
            freq = DataFrequency.year
        else:
            raise HTTPException(
                status_code=400,
                detail=NotSupportedFreqError().detail
            )

        # Валидация формата дат для определенной частотности
        self._validate_dates(sorted_ts, freq)

        return freq