import numpy as np

from src.core.domain.distributions import Distribution, DistributionServiceI, PDF, CDF


class DistributionFactory:
    registry: dict[str, type[DistributionServiceI]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(dist_class: type[DistributionServiceI]):
            cls.registry[name] = dist_class
            return dist_class

        return wrapper

    @classmethod
    def get_cdf(cls, x: list[float], distribution: Distribution, must_sort: bool) -> CDF:
        x = np.array(x)
        x = x[~np.isnan(x)]

        try:
            return cls.registry[distribution]().get_cdf(x, must_sort=must_sort)
        except KeyError:
            raise NotImplementedError(f"Распределение {distribution.name} пока не реализовано")

    @classmethod
    def get_pdf(cls, x: list[float], distribution: Distribution, must_sort: bool) -> PDF:
        x = np.array(x)
        x = x[~np.isnan(x)]

        try:
            return cls.registry[distribution]().get_pdf(x, must_sort=must_sort)
        except KeyError:
            raise NotImplementedError(f"Распределение {distribution.name} пока не реализовано")
