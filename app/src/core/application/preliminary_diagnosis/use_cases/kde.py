import numpy as np
from sklearn.neighbors import KernelDensity

from src.core.application.preliminary_diagnosis.schemas.kde import KdeParams, KdeResult, KDE, Histogram, KdeMethodUnion
from src.infrastructure.adapters.preliminary_diagnosis.kde_factory import KdeFactory


class KdeUC:
    def __init__(
            self,
            kde_factory: KdeFactory,
    ):
        self._kde_factory = kde_factory

    @staticmethod
    def auto_grid(x: np.array) -> np.ndarray:
        n = max(len(x), 10)
        num_points = int(np.clip(np.sqrt(n) * 50, 200, 2000))

        x_min, x_max = np.nanmin(x), np.nanmax(x)
        dx = x_max - x_min
        x_min -= dx * 0.05
        x_max += dx * 0.05

        grid = np.linspace(x_min, x_max, num_points)
        return grid

    @staticmethod
    def histogram(x, bins: int, density: bool) -> Histogram:
        counts, edges = np.histogram(x, bins=bins, density=density)
        centers = (edges[:-1] + edges[1:]) / 2
        width = np.diff(edges)
        return Histogram(
            counts=counts.tolist(),
            centers=centers.tolist(),
            width=width.tolist()
        )

    def kde_eval(self, x: np.array, grid: np.array, method: KdeMethodUnion) -> KDE:
        bandwidth = self._kde_factory.calculate_bandwidth(x, method)

        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(x.reshape(-1, 1))
        logdens = kde.score_samples(grid.reshape(-1, 1))
        dens = np.exp(logdens)
        return KDE(
            name=method.name,
            bandwidth=bandwidth,
            density=dens.tolist()
        )


    def execute(self, request: KdeParams) -> KdeResult:
        x = np.array(request.timeseries.values)

        histogram = self.histogram(x, bins=request.bins, density=request.density)
        grid = self.auto_grid(x)

        kde_results = []
        for method in request.methods:
            kde_results.append(self.kde_eval(x, grid, method))

        return KdeResult(histogram=histogram, kde_list=kde_results, grid=grid.tolist())