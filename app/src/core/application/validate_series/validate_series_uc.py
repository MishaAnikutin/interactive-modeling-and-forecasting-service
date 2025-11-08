from src.core.domain.validation import ValidationIssue

from .schemas import ValidationResponse, ValidationRequest

from src.infrastructure.factories.validation import ValidationVisitor
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class ValidateSeriesUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
        validation_visitor: ValidationVisitor,
    ):
        self._ts_adapter = ts_adapter
        self._validation_visitor = validation_visitor

    def execute(self, request: ValidationRequest) -> ValidationResponse:
        ts = self._ts_adapter.to_series(request.ts)

        issues: list[ValidationIssue] = self._validation_visitor.check(ts)

        return ValidationResponse(issues=issues)
