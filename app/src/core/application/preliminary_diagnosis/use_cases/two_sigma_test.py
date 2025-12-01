from fastapi import HTTPException
from src.core.application.preliminary_diagnosis.schemas.series_monitoring import (
    TwoSigmaTestResponse,
    TwoSigmaTestRequest,
)
from src.infrastructure.adapters.stat_tests.two_sigma import TwoSigmaTestAdapter
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class TwoSigmaTestUC:
    def __init__(
            self,
            ts_adapter: PandasTimeseriesAdapter,
            two_sigma_test_adapter: TwoSigmaTestAdapter
    ):
        self._ts_adapter = ts_adapter
        self._two_sigma_test_adapter = two_sigma_test_adapter

    def execute(self, request: TwoSigmaTestRequest) -> TwoSigmaTestResponse:
        timeseries = self._ts_adapter.to_series(request.timeseries)

        # try:
        results = self._two_sigma_test_adapter.perform_two_sigma_test(
            timeseries=timeseries,
            frequency=request.timeseries.data_frequency,
            date_boundary=request.date_boundary
        )
        # except Exception as exc:
        #     raise HTTPException(status_code=400, detail=str(exc))
        # else:
        return TwoSigmaTestResponse(results=results)