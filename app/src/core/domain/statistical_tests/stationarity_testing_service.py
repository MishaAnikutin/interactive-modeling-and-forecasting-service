from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from src.core.application.preliminary_diagnosis.schemas.stats_test import StatTestResult


class StationarityTestingService(ABC):
    def __init__(self, ts: pd.Series, alpha: float):
        self.ts = ts
        self.alpha = alpha

    @abstractmethod
    def apply(self, **kwargs) -> Optional[StatTestResult]:
        raise NotImplementedError