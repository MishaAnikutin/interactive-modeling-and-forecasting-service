import pandas as pd

from src.core.application.preprocessing.preprocess_scheme import DiffTransformation
from src.core.domain.preprocessing.service import PreprocessingServiceI
from src.infrastructure.adapters.preprocessing.preprocess_factory import PreprocessFactory


@PreprocessFactory.register(transform_type="diff")
class Diff(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: DiffTransformation) -> pd.Series:
        return ts