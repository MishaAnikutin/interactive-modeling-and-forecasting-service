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
    k: Optional[int] = Field(title="Число ближайших соседей для оценки", default=None, ge=1, le=1000)


class CrossValidationMethod(KdeMethod):
    name: Literal["cross validation"] = "cross validation"
    folds: int = Field(
        default=5,
        ge=2, le=100,
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
    bins: int = Field(title="Число интервалов", default=40, gt=0, lt=1000)
    density: bool = Field(
        description="Если True - возвращает плотность распределения, нормированную на 1. "
                    "Если False - абсолютные частоты (количество наблюдений в каждом бине).",
        default=True
    )
    methods: list[KdeMethodUnion] = Field(title="Список методов определения параметра сглаживания")


class KDE(BaseModel):
    """Данные для построения KDE"""
    density: list[float] = Field(..., title="Значения плотности")
    name: str = Field(..., title="Название метода")
    bandwidth: float = Field(..., title="Оптимальная ширина ядра")


class Histogram(BaseModel):
    """Данные для построения гистограммы"""
    centers: list[float] = Field(..., title="Центры столбцов")
    counts: list[float] = Field(..., title="Высоты столбцов / плотность")
    width: list[float] = Field(..., title="Ширина столбцов")


class KdeResult(BaseModel):
    """Данные для отрисовки KDE и гистограммы"""
    grid: list[float] = Field(..., title="Сетка X-координат")
    kde_list: list[KDE] = Field(..., title="Список результатов KDE")
    histogram: Histogram = Field(..., title="Данные гистограммы")

