from typing import List

import pandas as pd
from fastapi import HTTPException

from src.core.domain import Timeseries
from . import PandasTimeseriesAdapter
from .frequency_determiner import FrequencyDeterminer


class TimeseriesAlignment:
    def __init__(self):
        self._freq_determiner = FrequencyDeterminer()
        self._pandas_adapter = PandasTimeseriesAdapter()

    def compare(self, timeseries_list: List[Timeseries], target: Timeseries) -> pd.DataFrame:
        series_list = [self._pandas_adapter.to_series(target)]
        for ts_obj in timeseries_list:
            # Определяем тип временного ряда
            freq_type = self._freq_determiner.determine(ts_obj.dates)

            if freq_type != ts_obj.data_frequency:
                raise HTTPException(
                    detail=f"Не соответствует полученных тип ряда и заявленный для {ts_obj.name}. "
                           f"Определенное системой: {freq_type}, Заявленное: {ts_obj.data_frequency}",
                    status_code=400
                )
            if freq_type != target.data_frequency:
                raise HTTPException(
                    detail=f"Частотность экзогенной переменной {ts_obj.name} не соответствует частотности целевой переменной. "
                           f"Экзогенная переменная: {freq_type}, Целевая переменная: {target.data_frequency}",
                    status_code=400
                )

            # Создаем временной ряд
            series_list.append(self._pandas_adapter.to_series(ts_obj))

        # Объединяем ряды
        df = pd.concat(series_list, axis=1, join="inner")
        return df