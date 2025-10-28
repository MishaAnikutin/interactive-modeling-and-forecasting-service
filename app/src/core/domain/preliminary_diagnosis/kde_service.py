from abc import ABC, abstractmethod
import numpy as np

from src.core.application.preliminary_diagnosis.schemas.kde import KdeMethodUnion


class KdeServiceI(ABC):
    def __init__(self, ts: np.ndarray):
        self.ts_ = ts

    @abstractmethod
    def calculate_bandwidth(self, method: KdeMethodUnion) -> float:
        ...
