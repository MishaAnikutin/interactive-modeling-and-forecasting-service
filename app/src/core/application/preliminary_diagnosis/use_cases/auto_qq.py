from src.core.application.preliminary_diagnosis.schemas.auto_qq import AutoQQRequest, AutoQQResult
from src.core.application.preliminary_diagnosis.schemas.select_distribution import SelectDistResult, SelectDistRequest
from src.infrastructure.adapters.dist_fit.dist_fit import DistFit
from src.infrastructure.factories.distributions import DistributionFactory


class AutoQQplotUC:
    def __init__(self, select_dist: DistFit, dist_factory: DistributionFactory):
        self._dist_factory = dist_factory
        self._select_dist = select_dist

    def execute(self, request: AutoQQRequest) -> AutoQQResult:
        dist_fit_request = SelectDistRequest(
            timeseries=request.timeseries,
            method='parametric',
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
        x = request.timeseries.values

        ppf = self._dist_factory.get_ppf(x=x, distribution=best_dist)

        return AutoQQResult(
            theoretical_probs=ppf.y,
            empirical_probs=ppf.x,
            dist_name=best_dist,
        )
