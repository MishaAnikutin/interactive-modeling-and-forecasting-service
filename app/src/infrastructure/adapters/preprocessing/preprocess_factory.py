import pandas as pd

from src.core.application.preprocessing.preprocess_scheme import TransformationUnion
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
    def apply(cls, ts: pd.Series, transformation: TransformationUnion) -> pd.Series:
        return cls.registry[transformation.type]().apply(ts=ts, transformation=transformation)
