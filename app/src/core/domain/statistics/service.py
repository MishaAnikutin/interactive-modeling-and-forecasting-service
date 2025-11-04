from abc import ABC, abstractmethod
import numpy as np


class StatisticsServiceI(ABC):
    @abstractmethod
    def get_value(self, ts: np.ndarray) -> float:
        ...
