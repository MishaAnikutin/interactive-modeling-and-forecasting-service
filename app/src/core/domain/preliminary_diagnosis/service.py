from abc import ABC, abstractmethod
import numpy as np


class DistributionServiceI(ABC):
    @abstractmethod
    def get_theoretical_probs(self, ts: np.array) -> list[float]:
        ...
