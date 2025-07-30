from abc import ABC, abstractmethod
from typing import Tuple, Optional

import pandas as pd

from src.core.application.preprocessing.preprocess_scheme import TransformationUnion, PreprocessContext


class PreprocessingServiceI(ABC):
    @abstractmethod
    def apply(self, ts: pd.Series, transformation: TransformationUnion) -> Tuple[pd.Series, Optional[PreprocessContext]]:
        """
        Предобрабатывает ряд

        Если в процессе предобработки генерируется контекст необходимый для его
        обратного восстановления к исходному виду, добавляется PreprocessContext,
        иначе вторым аргументом возвращается None
        """
        ...

    @abstractmethod
    def inverse(self, ts: pd.Series, transformation: TransformationUnion) -> pd.Series:
        """Отменяет предобработку"""
        ...
