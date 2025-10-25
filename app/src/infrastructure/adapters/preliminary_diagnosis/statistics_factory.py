import numpy as np

from src.core.application.preliminary_diagnosis.schemas.statistics import StatisticsEnum
from src.core.domain.preliminary_diagnosis.statistics_service import StatisticsServiceI


class StatisticsFactory:
    registry: dict[str, type[StatisticsServiceI]] = {}

    @classmethod
    def register(cls, name: StatisticsEnum):
        def wrapper(stat_class: type[StatisticsServiceI]):
            cls.registry[name] = stat_class
            return stat_class

        return wrapper

    @classmethod
    def get_value(cls, ts: np.ndarray, statistic: StatisticsEnum) -> float:
        return cls.registry[statistic]().get_value(ts)
