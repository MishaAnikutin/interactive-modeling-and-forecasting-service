import abc
from typing import TypeAlias


PValue: TypeAlias = float
TestStatistic: TypeAlias = float


class StatTestService(abc.ABC):
    @abc.abstractmethod
    def calculate(self, **kwargs) -> tuple[TestStatistic, PValue]:
        ...
