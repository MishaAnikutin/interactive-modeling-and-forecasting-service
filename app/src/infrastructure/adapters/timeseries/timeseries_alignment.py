from typing import List

import pandas as pd
from fastapi import HTTPException

from src.core.application.building_model.errors.alignment import NotEqualToTargetError, NotEqualToExpectedError
from src.core.domain import Timeseries, DataFrequency
from . import PandasTimeseriesAdapter
from .frequency_determiner import FrequencyDeterminer


class TimeseriesAlignment:
    def __init__(self):
        self._freq_determiner = FrequencyDeterminer()
        self._pandas_adapter = PandasTimeseriesAdapter()

    def is_ts_freq_equal_to_expected(self, ts: Timeseries) -> DataFrequency:
        freq_type = self._freq_determiner.determine(ts.dates)
        if freq_type != ts.data_frequency:
            raise HTTPException(
                detail=NotEqualToExpectedError().detail,
                status_code=400
            )
        return freq_type

    def compare(self, timeseries_list: List[Timeseries], target: Timeseries) -> pd.DataFrame:
        target_data_frequency = self.is_ts_freq_equal_to_expected(target)
        series_list = [self._pandas_adapter.to_series(target)]
        for ts_obj in timeseries_list:
            freq_type = self.is_ts_freq_equal_to_expected(ts_obj)
            if freq_type != target.data_frequency:
                raise HTTPException(
                    detail=NotEqualToTargetError().detail,
                    status_code=400
                )
            # Создаем временной ряд
            series_list.append(self._pandas_adapter.to_series(ts_obj))

        # Объединяем ряды
        df = pd.concat(series_list, axis=1, join="inner")
        return df