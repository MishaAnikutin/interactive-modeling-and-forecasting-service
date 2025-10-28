from pydantic import BaseModel, Field

from src.core.domain import Timeseries
from typing import List, Literal, Union, Annotated, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator


class KdeMethod(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SilvermanMethod(KdeMethod):
    name: Literal["silverman"] = "silverman"


class ScottMethod(KdeMethod):
    name: Literal["scott"] = "scott"


class KnnMethod(KdeMethod):
    name: Literal["knn"] = "knn"
    k: Optional[int] = Field(title="Число ближайших соседей для оценки", default=None)


class CrossValidationMethod(KdeMethod):
    name: Literal["cross validation"] = "cross validation"
    folds: int = Field(
        default=5,
        ge=2,
        title="Число фолдов K-fold, используемое при подборе bandwidth"
    )


KdeMethodUnion = Annotated[
    Union[
        SilvermanMethod,
        ScottMethod,
        KnnMethod,
        CrossValidationMethod,
    ],
    Field(discriminator="name")
]


class KdeParams(BaseModel):
    timeseries: Timeseries
    bins: int = Field(title="Число интервалов", default=40)
    density: bool = Field(
        description="Если True - возвращает плотность распределения, нормированную на 1. "
                    "Если False - абсолютные частоты (количество наблюдений в каждом бине).",
        default=True
    )
    methods: list[KdeMethodUnion]


class KDE(BaseModel):
    density: list[float]
    name: str
    bandwidth: float

class Histogram(BaseModel):
    centers: list[float]
    counts: list[float]
    width: list[float]

class KdeResult(BaseModel):
    grid: list[float]
    kde_list: list[KDE]
    histogram: Histogram
