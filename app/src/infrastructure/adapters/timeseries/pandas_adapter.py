import pandas as pd

from src.core.domain import Timeseries


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
        return pd.Series(
            data=ts_obj.values, index=ts_obj.dates, name=ts_obj.name
        )

    @staticmethod
    def from_series(series: pd.Series) -> Timeseries:
        return Timeseries(
            name=series.name,
            dates=series.index.to_list(),
            values=series.values.to_list()
        )
