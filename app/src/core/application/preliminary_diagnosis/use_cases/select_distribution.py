import time

from src.core.application.preliminary_diagnosis.schemas.select_distribution import SelectDistRequest, \
    SelectDistResponse, SelectDistResult
from src.core.domain.distributions import PDF, CDF
from src.infrastructure.adapters.dist_fit.dist_fit import DistFit
from src.infrastructure.factories.distributions import DistributionFactory


class SelectDistUC:
    def __init__(self, select_dist: DistFit, dist_factory: DistributionFactory):
        self._select_dist = select_dist
        self._dist_factory = dist_factory

    def execute(self, request: SelectDistRequest) -> SelectDistResponse:
        best_dists: list[SelectDistResult] = self._select_dist.calculate(request)

        x = request.timeseries.values
        best_dist = best_dists[0].name
        must_sort = False

        pdf_y = self._dist_factory.get_pdf(x=x, distribution=best_dist, must_sort=must_sort)
        cdf_y = self._dist_factory.get_cdf(x=x, distribution=best_dist, must_sort=must_sort)

        best_dist_pdf = PDF(x=x, y=pdf_y)
        best_dist_cdf = CDF(x=x, y=cdf_y)

        return SelectDistResponse(
            best_dists=best_dists,
            best_dist_pdf=best_dist_pdf,
            best_dist_cdf=best_dist_cdf
        )
