from typing import List, Optional

import pandas as pd

from src.core.domain import Timeseries
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter


class ResidAnalysisInterface:
    def __init__(
            self,
            ts_aligner: TimeseriesAlignment,
            ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter
        self._ts_aligner = ts_aligner

    def _aligned_data(
            self,
            target: Timeseries,
            exog: Optional[List[Timeseries]]
    ) -> tuple[Timeseries, Optional[pd.DataFrame]]:
        self._ts_aligner.is_ts_freq_equal_to_expected(target)
        freq = target.data_frequency
        if exog is not None:
            df = self._ts_aligner.compare(
                timeseries_list=exog,
                target=target
            )

            target = df[target.name]
            if type(target) == pd.DataFrame:
                target = target.iloc[:, 0]
            target = self._ts_adapter.from_series(target, freq=freq)
            exog_df = df.drop(columns=[target.name])
            if exog_df.empty:
                exog_df = None
        else:
            exog_df = None

        return target, exog_df

