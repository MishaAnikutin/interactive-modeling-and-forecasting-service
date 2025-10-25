import numpy as np

from src.core.application.preliminary_diagnosis.schemas.kde import KdeEnum
from src.core.domain.preliminary_diagnosis.kde_service import KdeServiceI



class KdeFactory:
    registry: dict[str, type[KdeServiceI]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(kde_class: type[KdeServiceI]):
            cls.registry[name] = kde_class
            return kde_class

        return wrapper

    @classmethod
    def calculate_kde(cls, ts: np.array, kde_method: KdeEnum):
        return cls.registry[kde_method](ts).calculate_kde()
