from typing import Union, List
from pydantic import BaseModel, Field

from src.core.domain import Timeseries
from src.core.domain.distributions import EstimateDensity, Histogram, EstimateHistogramParams, EstimateDensityResult


class EstimateDensityParams(BaseModel):
    # https://www.geeksforgeeks.org/machine-learning/kernel-density-estimation/
    kernel: EstimateDensity.Kernel = Field(
        title="Тип ядра для оценки плотности распределения",
        description="Математически (в одномерном случае) - симметричная функция "
                    "K((x - x_i)/h), сумма которых представляет оценку плотности распределения в точке: "
                    "\[ \hat{f}(x) = \frac{1}{nh}\sum_{i=1}^{n} K(\frac{x - x_i}{h}) \]\n"
                    "Зависит от значений ряда (x) и ширины (bandwidth)",
        default=EstimateDensity.Kernel.gaussian
    )
    bandwidth: Union[float, EstimateDensity.BandwidthMethods] = Field(
        title="Ширина (Пропускная способность/уровень сглаживания), либо её метод нахождения",
        description="Чем больше, тем результат более сглаженный, чем меньше тем более зашумленный",
        default=1.0
    )

    algorithm: EstimateDensity.Algorithm = Field(
        title="Алгоритм расчета плотности",
        default=EstimateDensity.Algorithm.kd_tree
    )


# FIXME: Тут нехватает доменного анализа
class AutoEstimateDensityParams(BaseModel):
    """Оценка параметров KernelDensity на кросс-валидации"""

    # FIXME: например, тут можно было бы выбрать метод кросс-валидации. Не только KFold,
    #  а например ShuffleSplit, RepeatedKFold, RollingWindow, ExplainedWindow ...
    n_splits: int = Field(title='Число делений выборки на кросс-валидации', gt=0, le=1000, default=5)


class DistributionsRequest(BaseModel):
    timeseries: Timeseries
    histogram_params: EstimateHistogramParams
    density_params: List[EstimateDensityParams | AutoEstimateDensityParams]


class DistributionsResult(BaseModel):
    histogram: Histogram = Field(..., title="Данные гистограммы")
    density: list[EstimateDensityResult] = Field(..., title="Данные оценки плотности распределения")
