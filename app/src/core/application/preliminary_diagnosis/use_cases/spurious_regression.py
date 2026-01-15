from src.core.application.preliminary_diagnosis.schemas.spurious_regression import SpuriousRegressionRequest, \
    SpuriousRegressionResponse
from src.infrastructure.adapters.timeseries import TimeseriesAlignment, PandasTimeseriesAdapter
from src.infrastructure.interactors.spurious_regression_checker.checker import SpuriousRegressionChecker


class SpuriousRegressionUC:
    def __init__(self, ts_aligner: TimeseriesAlignment, ts_adapter: PandasTimeseriesAdapter, checker: SpuriousRegressionChecker):
        self._ts_aligner = ts_aligner
        self._ts_adapter = ts_adapter
        self._checker = checker

    def execute(self, request: SpuriousRegressionRequest) -> SpuriousRegressionResponse:
        df = self._ts_aligner.compare(
            target=request.dependent_variable,
            timeseries_list=request.explanatory_variable
        )

        X = df.drop(columns=[request.dependent_variable.name])
        y = df[request.dependent_variable.name]

        r2, dw, number_of_significant_coefs, is_spurious = self._checker.check(y, X, )

        return SpuriousRegressionResponse(
            r2=r2, dw=dw,
            number_of_significant_coefs=number_of_significant_coefs,
            is_spurious=is_spurious
        )
