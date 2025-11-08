from abc import ABC, abstractmethod
from typing import Optional

from src.core.domain import Timeseries
from .validation import ValidationIssue


class ValidationStrategyI(ABC):
    @abstractmethod
    def check(self, ts: Timeseries) -> Optional[ValidationIssue]:
        ...
