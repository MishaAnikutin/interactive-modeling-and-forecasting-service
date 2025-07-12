from abc import ABC, abstractmethod

import pandas as pd

from src.core.application.preprocessing.preprocess_scheme import TransformationUnion


class PreprocessingServiceI(ABC):
    @abstractmethod
    def apply(self, ts: pd.Series, transformation: TransformationUnion) -> pd.Series:
        pass