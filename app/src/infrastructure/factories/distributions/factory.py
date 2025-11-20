import numpy as np

from src.core.domain.distributions import Distribution, DistributionServiceI, PDF, CDF, PPF


class DistributionFactory:
    registry: dict[str, type[DistributionServiceI]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(dist_class: type[DistributionServiceI]):
            cls.registry[name] = dist_class
            return dist_class

        return wrapper

    @classmethod
    def get_cdf(cls, x: list[float], distribution: Distribution) -> CDF:
        x = np.array(x)
        x = x[~np.isnan(x)]
        x = np.sort(x)

        try:
            return cls.registry[distribution]().get_cdf(x)
        except KeyError:
            raise NotImplementedError(f"Распределение {distribution.name} пока не реализовано")

    @classmethod
    def get_pdf(cls, x: list[float], distribution: Distribution) -> PDF:
        x = np.array(x)
        x = x[~np.isnan(x)]
        x = np.sort(x)

        try:
            return cls.registry[distribution]().get_pdf(x)
        except KeyError:
            raise NotImplementedError(f"Распределение {distribution.name} пока не реализовано")

    @classmethod
    def get_ppf(cls, x: list[float], distribution: Distribution) -> PPF:
        x = np.array(x)
        x = x[~np.isnan(x)]
        x = np.sort(x)

        try:
            return cls.registry[distribution]().get_ppf(x)
        except KeyError:
            raise NotImplementedError(f"Распределение {distribution.name} пока не реализовано")


    @classmethod
    def get_quantile(cls, x: list[float], q: float, distribution: Distribution) -> float:
        x = np.array(x)
        x = x[~np.isnan(x)]
        x = np.sort(x)

        try:
            return cls.registry[distribution]().get_quantile(x, q)
        except KeyError:
            raise NotImplementedError(f"Распределение {distribution.name} пока не реализовано")

