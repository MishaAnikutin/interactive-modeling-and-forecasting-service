import numpy as np

from src.core.application.preliminary_diagnosis.schemas.pp_plot import DistributionEnum
from src.core.domain.preliminary_diagnosis.service import DistributionServiceI


class PPplotFactory:
    registry: dict[str, type[DistributionServiceI]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(dist_class: type[DistributionServiceI]):
            cls.registry[name] = dist_class
            return dist_class

        return wrapper

    @classmethod
    def get_theoretical_probs(cls, ts: np.array, distribution: DistributionEnum) -> list[float]:
        return cls.registry[distribution.name]().get_theoretical_probs(ts=ts)
