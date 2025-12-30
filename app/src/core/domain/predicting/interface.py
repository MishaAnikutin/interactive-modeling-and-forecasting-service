from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from src.core.domain import DataFrequency, FitParams


class BasePredictor(ABC):
    @abstractmethod
    def execute(
            self,
            model_weight: bytes,
            fit_params: FitParams,
            data_frequency: DataFrequency,
            target: pd.Series,
            exog_df: Optional[pd.DataFrame]
    ) -> tuple[pd.Series, pd.Series]:
        pass