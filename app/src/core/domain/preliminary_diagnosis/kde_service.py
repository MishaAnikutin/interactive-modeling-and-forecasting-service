from abc import ABC, abstractmethod
import numpy as np


class KdeServiceI(ABC):
    def __init__(self, ts: np.array):
        self.ts_ = ts

    def get_x_grid(self) -> np.ndarray:
        x_min, x_max = self.ts_.min(), self.ts_.max()
        x_range = x_max - x_min
        x_grid = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)
        return x_grid

    @abstractmethod
    def calculate_kde(self) -> list[float]:
        ...
