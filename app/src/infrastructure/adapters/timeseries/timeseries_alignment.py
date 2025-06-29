from datetime import datetime, timedelta
from typing import List

import pandas as pd

from src.core.domain import Timeseries
from .frequency_determiner import FrequencyDeterminer


class TimeseriesAlignment:
    def __init__(self):
        self._freq_determiner = FrequencyDeterminer()

    def _convert_date_(self, dt: datetime, freq_type: str) -> datetime:
        """Преобразует дату в соответствии с типом временного ряда"""
        if freq_type == "M":
            if dt.day != 1:
                # Переносим на первый день следующего месяца
                next_month = dt.replace(day=28) + timedelta(
                    days=4
                )  # Переход на следующий месяц
                return next_month.replace(day=1)
            return dt
        elif freq_type == "H":
            next_day = dt + timedelta(days=1)
            return next_day.replace(hour=0, minute=0, second=0, microsecond=0)
        else:  # daily
            return dt

    def compare(self, timeseries_list: List[Timeseries]) -> pd.DataFrame:
        series_list = []
        for ts_obj in timeseries_list:
            # Определяем тип временного ряда
            freq_type = self._freq_determiner.determine(ts_obj.dates)

            # Применяем преобразование дат
            converted_dates = [
                self._convert_date_(dt, freq_type.value) for dt in ts_obj.dates
            ]

            # Создаем временной ряд
            series = pd.Series(
                ts_obj.values, index=pd.DatetimeIndex(converted_dates), name=ts_obj.name
            )
            series_list.append(series)

        # Объединяем ряды
        df = pd.concat(series_list, axis=1, join="inner")

        # Форматируем даты в единый строковый формат
        df.index = df.index.strftime("%d.%m.%Y %H:%M:%S")

        return df