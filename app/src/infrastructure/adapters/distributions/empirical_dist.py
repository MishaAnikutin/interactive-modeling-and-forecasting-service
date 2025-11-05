from src.core.domain.distributions import PDF, CDF
import numpy as np


# TODO: Подумать, не надо ли это все в Histogram было делать
class EmpiricalDistribution:
    def get_pdf(self, x: list[float]) -> PDF:
        x_array = np.array(x)
        hist, bin_edges = np.histogram(x_array, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return PDF(x=bin_centers.tolist(), y=hist.tolist())

    def get_cdf(self, x: list[float]) -> CDF:
        x_sorted = np.sort(x)
        cdf_y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        return CDF(x=x_sorted.tolist(), y=cdf_y.tolist())
