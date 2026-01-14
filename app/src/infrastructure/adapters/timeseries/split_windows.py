from datetime import date

import pandas as pd

from src.core.domain import Timeseries, WindowsForecast, FitParams, DataFrequency
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit, PandasTimeseriesAdapter


class WindowSplitter:
    def __init__(
            self,
            splitter: TimeseriesTrainTestSplit,
            pandas_adapter: PandasTimeseriesAdapter,
    ):
        self.splitter = splitter
        self.pandas_adapter = pandas_adapter

    def split_single(self, ts: pd.Series, boundary: date):
        pd_boundary = pd.Timestamp(boundary)
        if not isinstance(ts.index, pd.DatetimeIndex):
            ts = ts.copy()
            ts.index = pd.to_datetime(ts.index)
        idx = ts.index

        left_mask = idx <= pd_boundary
        right_mask = idx > pd_boundary

        left = ts.loc[left_mask]
        right = ts.loc[right_mask]

        return left, right

    def from_pandas(self, ts_list: list[pd.Series], freq) -> list[Timeseries]:
        return [self.pandas_adapter.from_series(series=ts, freq=freq) for ts in ts_list]

    def split(
            self,
            forecasts: list[Timeseries],
            fit_params: FitParams,
            last_date: date,
            freq: DataFrequency
    ) -> WindowsForecast:
        train_forecasts = []
        val_forecasts = []
        test_forecasts = []
        out_of_sample_forecasts = []

        for forecast in forecasts:
            pandas_forecast = self.pandas_adapter.to_series(forecast)
            fcst_train, fcst_val, fcst_test = self.splitter.split_ts(
                ts=pandas_forecast,
                train_boundary=fit_params.train_boundary,
                val_boundary=fit_params.val_boundary
            )
            if not fcst_train.empty:
                fcst_train, fcst_out = self.split_single(ts=fcst_train, boundary=last_date)
                if not fcst_out.empty:
                    out_of_sample_forecasts.append(fcst_out)
                if not fcst_train.empty:
                    train_forecasts.append(fcst_train)
            if not fcst_val.empty:
                fcst_val, fcst_out = self.split_single(ts=fcst_val, boundary=last_date)
                if not fcst_out.empty:
                    out_of_sample_forecasts.append(fcst_out)
                if not fcst_val.empty:
                    val_forecasts.append(fcst_val)
            if not fcst_test.empty:
                fcst_test, fcst_out = self.split_single(ts=fcst_test, boundary=last_date)
                if not fcst_out.empty:
                    out_of_sample_forecasts.append(fcst_out)
                if not fcst_test.empty:
                    test_forecasts.append(fcst_test)

        return WindowsForecast(
            train_forecasts=self.from_pandas(train_forecasts, freq),
            val_forecasts=self.from_pandas(val_forecasts, freq),
            test_forecasts=self.from_pandas(test_forecasts, freq),
            out_of_sample_forecasts=self.from_pandas(out_of_sample_forecasts, freq),
        )