import numpy as np

from src.core.domain.distributions import Histogram


class HistogramEstimator:
    def eval(self, values: list[float], bins: int, is_density: bool) -> Histogram:
        x = values

        counts, edges = np.histogram(x, bins=bins, density=is_density)
        centers = (edges[:-1] + edges[1:]) / 2
        width = np.diff(edges)

        return Histogram(
            counts=counts.tolist(),
            centers=centers.tolist(),
            width=width.tolist()
        )
