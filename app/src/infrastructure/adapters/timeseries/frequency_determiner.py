from datetime import datetime, timedelta
from typing import List

from src.core.domain import DataFrequency


class FrequencyDeterminer:
    """
    Класс-утилита, определяющий дискретность временного ряда.
    Алгоритм очень «инженерный» и опирается только на минимальный
    шаг между соседними метками.
    """

    # Границы (в секундах) между разными частотами
    _ONE_MINUTE = 60
    _ONE_HOUR = 60 * 60
    _ONE_DAY = 24 * _ONE_HOUR
    _ONE_MONTH = 28 * _ONE_DAY          # ~ 1 месяц
    _ONE_QUARTER = 80 * _ONE_DAY        # ~ квартал
    _ONE_YEAR = 300 * _ONE_DAY          # ~ год

    @classmethod
    def determine(cls, timestamps: List[datetime]) -> DataFrequency:
        """
        Определяет частоту временного ряда.
        Возвращает значение перечисления DataFrequency.
        """
        if not timestamps:
            # Пустой ряд — наиболее частый вариант «по умолчанию»
            return DataFrequency.day

        # Если присутствует время, отличное от 00:00:00,
        # значит, данные как минимум почасовые (или чаще).

        # Сортируем для корректного поиска минимального шага
        sorted_ts = sorted(timestamps)
        if len(sorted_ts) == 1:
            # Одиночная точка. Считаем, что это «месячное» наблюдение
            return DataFrequency.month

        # Минимальный шаг между соседними метками
        min_delta: timedelta = min(
            (sorted_ts[i] - sorted_ts[i - 1]) for i in range(1, len(sorted_ts))
        )
        min_seconds = min_delta.total_seconds()

        # ───────────────────────────────────────────────────────
        # Логика классификации
        # ───────────────────────────────────────────────────────
        if min_seconds < cls._ONE_MINUTE:
            return DataFrequency.minute

        if min_seconds < cls._ONE_HOUR:
            # Шаг ≥ 1 мин и < 1 ч
            return DataFrequency.minute

        if min_seconds < cls._ONE_DAY:
            return DataFrequency.hour

        if min_seconds < cls._ONE_MONTH:
            return DataFrequency.day

        if min_seconds < cls._ONE_QUARTER:
            return DataFrequency.month

        if min_seconds < cls._ONE_YEAR:
            return DataFrequency.quart

        return DataFrequency.year