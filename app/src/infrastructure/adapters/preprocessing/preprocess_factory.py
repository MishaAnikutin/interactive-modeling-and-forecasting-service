from typing import Tuple, Optional

import pandas as pd

from src.core.application.preprocessing.preprocess_scheme import (
    TransformationUnion,
    PreprocessContext,
    InverseTransformationUnion
)
from src.core.domain.preprocessing.service import PreprocessingServiceI


class PreprocessFactory:
    registry: dict[str, type[PreprocessingServiceI]] = {}

    @classmethod
    def register(cls, transform_type: str):
        def wrapper(preprocess_class: type[PreprocessingServiceI]):
            cls.registry[transform_type] = preprocess_class
            return preprocess_class

        return wrapper

    @classmethod
    def apply(cls, ts: pd.Series, transformation: TransformationUnion) -> Tuple[pd.Series, Optional[PreprocessContext]]:
        """Предобрабатывает ряд и возвращает контекст предобработки, если он есть"""

        return cls.registry[transformation.type]().apply(ts=ts, transformation=transformation)

    @classmethod
    def inverse(cls, ts: pd.Series, transformation: InverseTransformationUnion) -> pd.Series:
        return cls.registry[transformation.type]().inverse(ts=ts, transformation=transformation)
