from pydantic import BaseModel

from src.core.domain import Timeseries
from enum import Enum


class KdeEnum(str, Enum):
    silverman = 'silverman'
    scott = 'scott'
    knn = 'knn'
    cross_validation = 'cross_validation'


class KdeParams(BaseModel):
    timeseries: Timeseries
    bins: int
    kde_method: KdeEnum


class KDE(BaseModel):
    x_grid: list[float]
    bandwidth: float

class Histogram(BaseModel):
    bin_edges: list[float]
    density: list[float]
    bin_centers: list[float]

class KdeResult(BaseModel):
    kde: KDE
    histogram: Histogram
