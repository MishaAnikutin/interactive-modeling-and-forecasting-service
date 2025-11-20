from src.core.application.preliminary_diagnosis.schemas.select_distribution import SelectDistRequest, \
    SelectDistResponse, SelectDistResult
from src.infrastructure.adapters.dist_fit.dist_fit import DistFit
from src.infrastructure.adapters.distributions import EmpiricalDistribution, HistogramEstimator
from src.infrastructure.factories.distributions import DistributionFactory


class SelectDistUC:
    def __init__(
            self,
            select_dist: DistFit,
            dist_factory: DistributionFactory,
            empirical_dist: EmpiricalDistribution,
            histogram_estimator: HistogramEstimator
    ):
        self._select_dist = select_dist
        self._dist_factory = dist_factory
        self._empirical_dist = empirical_dist
        self._histogram_estimator = histogram_estimator

    def execute(self, request: SelectDistRequest) -> SelectDistResponse:
        best_dists: list[SelectDistResult] = self._select_dist.calculate(request)

        x = request.timeseries.values
        best_dist = best_dists[0].name

        best_dist_theoretical_pdf = self._dist_factory.get_pdf(x=x, distribution=best_dist)
        best_dist_theoretical_cdf = self._dist_factory.get_cdf(x=x, distribution=best_dist)

        empirical_pdf = self._histogram_estimator.eval(values=x, bins=request.bins, is_density=True)
        empirical_cdf = self._empirical_dist.get_cdf(x=x)

        p_05 = self._dist_factory.get_quantile(x=x, q=0.05, distribution=best_dist)
        p_95 = self._dist_factory.get_quantile(x=x, q=0.95, distribution=best_dist)

        return SelectDistResponse(
            p_05=p_05, p_95=p_95,
            best_dists=best_dists,
            best_dist_theoretical_pdf=best_dist_theoretical_pdf,
            best_dist_theoretical_cdf=best_dist_theoretical_cdf,
            empirical_pdf=empirical_pdf,
            empirical_cdf=empirical_cdf
        )
