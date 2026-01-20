from abc import ABC, abstractmethod
import numpy as np

from src.core.application.preliminary_diagnosis.schemas.statistics import StatisticResult


class StatisticsServiceI(ABC):
    @abstractmethod
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        ...
