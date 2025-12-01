from fastapi import HTTPException
from src.core.application.preliminary_diagnosis.schemas.series_monitoring import (
    FisherTestResponse,
    FisherTestRequest,
)
from src.infrastructure.adapters.stat_tests.fisher import FisherTestAdapter
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class FisherTestUC:
    def __init__(
            self,
            ts_adapter: PandasTimeseriesAdapter,
            fisher_test_adapter: FisherTestAdapter
    ):
        self._ts_adapter = ts_adapter
        self._fisher_test_adapter = fisher_test_adapter

    def execute(self, request: FisherTestRequest) -> FisherTestResponse:
        timeseries = self._ts_adapter.to_series(request.timeseries)

        try:
            results = self._fisher_test_adapter.perform_fisher_test(
                timeseries=timeseries,
                frequency=request.timeseries.data_frequency,
                date_boundary=request.date_boundary,
                alpha=request.alpha
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        else:
            return FisherTestResponse(results=results)
