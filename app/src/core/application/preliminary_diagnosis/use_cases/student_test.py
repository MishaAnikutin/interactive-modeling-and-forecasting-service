from fastapi import HTTPException

from src.core.application.preliminary_diagnosis.schemas.series_monitoring import (
    StudentTestResponse,
    StudentTestRequest,
)
from src.infrastructure.adapters.stat_tests.student import StudentTestAdapter
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class StudentTestUC:
    def __init__(
            self,
            ts_adapter: PandasTimeseriesAdapter,
            student_test_adapter: StudentTestAdapter
    ):
        self._ts_adapter = ts_adapter
        self._student_test_adapter = student_test_adapter

    def execute(self, request: StudentTestRequest) -> StudentTestResponse:

        timeseries = self._ts_adapter.to_series(request.timeseries)

        try:
            results = self._student_test_adapter.perform_student_test(
                timeseries=timeseries,
                frequency=request.timeseries.data_frequency,
                date_boundary=request.date_boundary,
                equal_var=request.equal_var,
                alpha=request.alpha
            )

        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        else:
            return StudentTestResponse(results=results)
