from typing import List

import pandas as pd

from src.core.domain import Timeseries, DataFrequency

def process_float(val):
    if pd.isna(val):
        return None
    val = float(val)
    if val == float('nan'):
        return None
    elif val == float('-inf') or val == float('inf'):
        return None
    return val

class PandasTimeseriesAdapter:
    @staticmethod
    def to_dataframe(ts: Timeseries) -> pd.DataFrame:
        return pd.DataFrame({ts.name: ts.values}, index=pd.to_datetime(ts.dates))

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> Timeseries:
        name = df.columns[0]
        return Timeseries(
            name=name, dates=df.index.to_list(), values=df[name].to_list()
        )

    @staticmethod
    def to_series(ts_obj: Timeseries) -> pd.Series:
        dates = pd.to_datetime(ts_obj.dates)
        return pd.Series(
            data=ts_obj.values, index=dates, name=ts_obj.name
        )

    @staticmethod
    def from_series(series: pd.Series, freq: DataFrequency) -> Timeseries:
        res = Timeseries(
            name=series.name,
            dates=[d for d in series.index],
            values=[process_float(v) for v in series.values],
            data_frequency=freq
        )
        return res

    def to_dataframe_from_list(self, list_ts: List[Timeseries]) -> pd.DataFrame:
        res_df = self.to_dataframe(list_ts[0])
        for ts in list_ts[1:]:
            res_df = pd.concat([res_df, self.to_dataframe(ts)], axis=1)
        return res_df

    def from_dataframe_to_list(self, df: pd.DataFrame, freq):
        res_list = []
        for col in df.columns:
            res_list.append(self.from_series(df[col], freq))
        return res_list

