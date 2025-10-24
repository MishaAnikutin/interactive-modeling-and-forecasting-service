from src.core.application.preliminary_diagnosis.schemas.qq import QQResult, QQParams
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class MeanUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: QQParams) -> QQResult:
        ...
