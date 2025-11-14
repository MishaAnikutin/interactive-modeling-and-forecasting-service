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
        )

        return PPResult(
            theoretical_probs=cdf.y,
            empirical_probs=(np.arange(1, n + 1) - 0.5) / n,
        )
