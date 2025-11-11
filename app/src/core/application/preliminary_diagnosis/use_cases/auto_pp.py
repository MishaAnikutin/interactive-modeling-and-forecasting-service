from src.core.application.preliminary_diagnosis.schemas.auto_pp import AutoPPRequest
from src.core.application.preliminary_diagnosis.schemas.pp_plot import PPResult
from src.core.application.preliminary_diagnosis.schemas.select_distribution import SelectDistResult, SelectDistRequest
from src.infrastructure.adapters.dist_fit.dist_fit import DistFit
from src.infrastructure.factories.distributions import DistributionFactory
import numpy as np


class AutoPPplotUC:
    def __init__(self, select_dist: DistFit, dist_factory: DistributionFactory):
        self._dist_factory = dist_factory
        self._select_dist = select_dist

    def execute(self, request: AutoPPRequest) -> PPResult:
        dist_fit_request = SelectDistRequest(
            timeseries=request.timeseries,
            method=request.method,
            distribution=[
                "norm", "expon", "pareto",
                "dweibull", "t",
                "genextreme", "gamma", "lognorm",
                "beta", "uniform", "loggamma"
            ],
            statistics='RSS',
            bins=40
        )
        best_dists: list[SelectDistResult] = self._select_dist.calculate(request=dist_fit_request)
        best_dist = best_dists[0].name

        n = len(request.timeseries.values)

        best_dist_theoretical_cdf = self._dist_factory.get_cdf(
            x=request.timeseries.values,
            distribution=best_dist,
            must_sort=True
        )

        theoretical_probs = best_dist_theoretical_cdf.y
        empirical_probs = (np.arange(1, n + 1) - 0.5) / n

        return PPResult(
            theoretical_probs=theoretical_probs,
            empirical_probs=empirical_probs.tolist(),
        )
