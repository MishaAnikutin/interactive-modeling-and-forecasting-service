from src.core.application.preliminary_diagnosis.schemas.mode_value import ModeParams, ModeResult
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
import pandas as pd


class ModeUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: ModeParams) -> ModeResult:
        series = pd.Series(request.timeseries.values)
        value_counts = series.value_counts()
        mode_value = value_counts.index[0]
        return ModeResult(value=round(mode_value, 4))
