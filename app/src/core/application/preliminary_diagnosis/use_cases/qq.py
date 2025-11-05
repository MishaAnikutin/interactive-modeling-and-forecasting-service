from src.infrastructure.factories.distributions import DistributionFactory
from src.core.application.preliminary_diagnosis.schemas.qq import QQResult, QQParams


class QQplotUC:
    def __init__(
        self,
        dist_factory: DistributionFactory,
    ):
        self._dist_factory = dist_factory

    def execute(self, request: QQParams) -> QQResult:
        pdf = self._dist_factory.get_pdf(
            x=request.timeseries.values,
            distribution=request.theoretical_dist,
            must_sort=True
        )

        theoretical_quantiles = pdf.y

        sample_sorted = sorted(request.timeseries.values)

        return QQResult(
            normal_values=theoretical_quantiles,
            data_values=sample_sorted
        )
