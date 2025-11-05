from src.core.application.preliminary_diagnosis.schemas.pp_plot import PPplotParams, PPResult
from src.infrastructure.factories.distributions import DistributionFactory

# FIXME
import numpy as np


class PPplotUC:
    def __init__(
        self,
        dist_factory: DistributionFactory,
    ):
        self._dist_factory = dist_factory

    def execute(self, request: PPplotParams) -> PPResult:
        n = len(request.timeseries.values)

        cdf = self._dist_factory.get_cdf(
            x=request.timeseries.values,
            distribution=request.distribution,
            must_sort=True
        )

        theoretical_probs = cdf.y

        # FIXME
        empirical_probs = np.arange(1, n + 1) / (n + 1)

        perfect_line_x = [0, 1]
        perfect_line_y = perfect_line_x

        return PPResult(
            theoretical_probs=theoretical_probs,
            empirical_probs=empirical_probs.tolist(),
            perfect_line_x=perfect_line_x,
            perfect_line_y=perfect_line_y,
        )
