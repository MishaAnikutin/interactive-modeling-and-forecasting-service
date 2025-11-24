
from src.core.application.preliminary_diagnosis.schemas.kde import (
    DistributionsRequest,
    AutoEstimateDensityParams,
    DistributionsResult
)
from src.core.domain.distributions import EstimateDensityResult, Histogram

from src.infrastructure.adapters.distributions import DensityEstimator, HistogramEstimator


class EstimateDistributionsUC:
    def __init__(
            self,
            density_estimator: DensityEstimator,
            histogram_estimator: HistogramEstimator
    ):
        self._density_estimator = density_estimator
        self._histogram_estimator = histogram_estimator

    def execute(self, request: DistributionsRequest) -> DistributionsResult:
        values: list[float] = request.timeseries.values

        histogram: Histogram = self._histogram_estimator.eval(
            values=values,
            bins=request.histogram_params.bins,
            is_density=request.histogram_params.is_density
        )

        estimate_density_results: list[EstimateDensityResult] = []
        for params in request.density_params:
            match params:
                # Если автоматически получаем параметры, то оцениваем их на кросс-валидации
                case AutoEstimateDensityParams():
                    density = self._density_estimator.auto_eval(values=values, n_splits=params.n_splits, step=params.step)

                # Иначе - по введенным
                case _:
                    density = self._density_estimator.eval(
                        values=values,
                        kernel=params.kernel,
                        algorithm=params.algorithm,
                        bandwidth=params.bandwidth,
                        step=params.step
                    )

            estimate_density_results.append(density)


        return DistributionsResult(
            histogram=histogram,
            density=estimate_density_results,
        )
