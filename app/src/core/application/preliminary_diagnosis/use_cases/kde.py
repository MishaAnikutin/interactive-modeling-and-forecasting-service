import numpy as np

from src.core.application.preliminary_diagnosis.schemas.kde import KdeParams, KdeResult, KDE, Histogram
from src.infrastructure.adapters.preliminary_diagnosis.kde_factory import KdeFactory


class KdeUC:
    def __init__(
            self,
            kde_factory: KdeFactory,
    ):
        self._kde_factory = kde_factory

    def execute(self, request: KdeParams) -> KdeResult:
        data = np.asarray(request.timeseries.values)

        hist_counts, bin_edges = np.histogram(data, bins=request.bins, density=True)
        x_grid, density, bandwidth = self._kde_factory.calculate_kde(data, request.kde_method)

        return KdeResult(
            kde=KDE(
                x_grid=x_grid.tolist(),
                bandwidth=bandwidth,
            ),
            histogram=Histogram(
                bin_edges=bin_edges.tolist(),
                density=density.tolist(),
                bin_centers=((bin_edges[:-1] + bin_edges[1:]) / 2).tolist(),
            )
        )
