from enum import Enum, StrEnum

from pydantic import BaseModel, Field


class EstimateDensity:
    """
    Непараметрический метод нахождения теоретической функции плотности распределения.
    """

    class BandwidthMethods(StrEnum):
        scott: str = 'scott'
        silverman: str = 'silverman'

    class Algorithm(StrEnum):
        kd_tree: str = 'kd_tree'
        ball_tree: str = 'ball_tree'
        # 'auto' решил не добавлять, т.к. там примитивная логика

    class Kernel(StrEnum):
        gaussian: str = 'gaussian'
        tophat: str = 'tophat'
        epanechnikov: str = 'epanechnikov'
        exponential: str = 'exponential'
        linear: str = 'linear'
        cosine: str = 'cosine'


class PDF(BaseModel):
    x: list[float] = Field(..., description="Сетка значений X")
    y: list[float] = Field(..., title="Список значений плотности распределения вероятности")


class CDF(BaseModel):
    x: list[float] = Field(..., description="Сетка значений X")
    y: list[float] = Field(..., title="Список значений кумулятивной функции распределения")


class PPF(BaseModel):
    x: list[float] = Field(..., description="Сетка значений X")
    y: list[float] = Field(..., title="Список значений поточечной обратной cdf")


class DensityKernelResult(BaseModel):
    score: float
    bandwidth: float
    kernel: EstimateDensity.Kernel
    algorithm: EstimateDensity.Algorithm


class EstimateDensityResult(BaseModel):
    density: PDF
    result:  DensityKernelResult
