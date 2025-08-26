from typing import List, Optional

import pandas as pd
from fastapi import HTTPException

from src.core.application.building_model.errors.alignment import NotEqualToExpectedError
from src.core.domain import Timeseries, DataFrequency
from src.core.domain.model.model_data import ModelData
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

    def align(self, model_data: ModelData) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        self.is_ts_freq_equal_to_expected(model_data.dependent_variables)

        if model_data.explanatory_variables:
            df = self.compare(
                timeseries_list=model_data.explanatory_variables,
                target=model_data.dependent_variables
            )

            target = df[model_data.dependent_variables.name]
            if type(target) == pd.DataFrame:
                target = target.iloc[:, 0]
            exog_df = df.drop(columns=[model_data.dependent_variables.name])
            if exog_df.empty:
                exog_df = None
        else:
            target = self._pandas_adapter.to_series(model_data.dependent_variables)
            exog_df = None

        return target, exog_df

    def compare(self, timeseries_list: List[Timeseries], target: Timeseries) -> pd.DataFrame:
        self.is_ts_freq_equal_to_expected(target)
        series_list = [self._pandas_adapter.to_series(target)]
        for ts_obj in timeseries_list:
            freq_type = self.is_ts_freq_equal_to_expected(ts_obj)
            if freq_type != target.data_frequency:
                raise HTTPException(
                    # detail=NotEqualToTargetError().detail,  FIXME
                    detail=NotEqualToExpectedError().detail,
                    status_code=400
                )
            # Создаем временной ряд
            series_list.append(self._pandas_adapter.to_series(ts_obj))

        # Объединяем ряды
        df = pd.concat(series_list, axis=1, join="inner")
        return df