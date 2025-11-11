from src.core.application.preliminary_diagnosis.schemas.auto_qq import AutoQQRequest
from src.core.application.preliminary_diagnosis.schemas.qq import QQResult
from src.core.application.preliminary_diagnosis.schemas.select_distribution import SelectDistResult, SelectDistRequest
from src.infrastructure.adapters.dist_fit.dist_fit import DistFit
from src.infrastructure.factories.distributions import DistributionFactory


class AutoQQplotUC:
    def __init__(self, select_dist: DistFit, dist_factory: DistributionFactory):
        self._dist_factory = dist_factory
        self._select_dist = select_dist

    def execute(self, request: AutoQQRequest) -> QQResult:
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
        x = request.timeseries.values

        best_dist_theoretical_pdf = self._dist_factory.get_pdf(
            x=x,
            distribution=best_dist,
            must_sort=True
        )

        sample_sorted = sorted(x)

        return QQResult(
            theoretical_probs=best_dist_theoretical_pdf.y,
            empirical_probs=sample_sorted,
        )
