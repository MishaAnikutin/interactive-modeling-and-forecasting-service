from datetime import datetime, time, timedelta
from typing import List


class FrequencyDeterminer:
    @staticmethod
    def determine(timestamps: List[datetime]) -> str:
        """
        Определяет тип временного ряда на основе временных меток.
        Возвращает:
            'monthly' - месячные данные
            'daily' - дневные данные
            'hourly' - часовые данные
        """
        if len(timestamps) == 0:
            return "daily"  # По умолчанию для пустых рядов

        # Проверяем наличие времени в данных (не только полночь)
        has_time = any(ts.time() != time(0, 0) for ts in timestamps)
        if has_time:
            return "hourly"

        if len(timestamps) == 1:
            return "monthly"  # По умолчанию для коротких рядов

        # Анализируем интервалы между метками
        sorted_ts = sorted(timestamps)
        min_diff = min(
            (sorted_ts[i] - sorted_ts[i - 1]) for i in range(1, len(sorted_ts))
        )

        # Преобразуем в дни для анализа
        min_diff_days = min_diff.total_seconds() / (24 * 3600)

        if min_diff_days >= 15:  # Большие интервалы - месячные данные
            return "monthly"
        return "daily"  # Короткие интервалы - дневные данные
