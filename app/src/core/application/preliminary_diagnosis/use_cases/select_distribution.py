import numpy as np

from src.core.application.preliminary_diagnosis.schemas.qq import QQResult
from src.core.application.preliminary_diagnosis.schemas.select_distribution import SelectDistRequest, \
    SelectDistResponse, SelectDistResult
from src.infrastructure.adapters.dist_fit.dist_fit import DistFit


class SelectDistUC:
    def __init__(self, select_dist: DistFit):
        self._select_dist = select_dist
        # self._calculate_qq = calculate_qq

    def execute(self, request: SelectDistRequest) -> SelectDistResponse:
        # FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME
        best_dists, best_dist, best_params = self._select_dist.calculate(request)

        best_dists: list[SelectDistResult]

        qq_result = self.qqplot_generic(x=request.timeseries.values, best_dist=best_dist, best_params=best_params)

        return SelectDistResponse(best_dists=best_dists, qq_result=qq_result)

    @staticmethod
    def qqplot_generic(x, best_dist, best_params) -> QQResult:
        # FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME МНЕ ПЛОХО

        x = np.asarray(x)
        x_sorted = np.sort(x)
        n = len(x_sorted)
        probs = (np.arange(1, n + 1) - 0.5) / n

        theo_quants = best_dist.ppf(probs, *best_params)
        return QQResult(data_values=list(x_sorted), normal_values=list(theo_quants))
