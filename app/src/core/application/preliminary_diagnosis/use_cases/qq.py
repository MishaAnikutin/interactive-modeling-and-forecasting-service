from src.infrastructure.factories.distributions import DistributionFactory
from src.core.application.preliminary_diagnosis.schemas.qq import QQResult, QQParams


class QQplotUC:
    def __init__(
        self,
        dist_factory: DistributionFactory,
    ):
        self._dist_factory = dist_factory

    def execute(self, request: QQParams) -> QQResult:
        ppf = self._dist_factory.get_ppf(
            x=request.timeseries.values,
            distribution=request.distribution
        )

        return QQResult(
            theoretical_probs=ppf.y,
            empirical_probs=ppf.x
        )
