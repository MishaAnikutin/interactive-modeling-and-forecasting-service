from typing import Union

from pydantic import BaseModel, model_validator, Field
from enum import StrEnum

from src.core.application.preliminary_diagnosis.schemas.qq import QQResult
from src.core.domain import Timeseries


class Distribution(StrEnum):
    norm: str = "norm"
    expon: str = "expon"
    pareto: str = "pareto"
    dweibull: str = "dweibull"
    t: str = "t"
    genextreme: str = "genextreme"
    gamma: str = "gamma"
    lognorm: str = "lognorm"
    beta: str = "beta"
    uniform: str = "uniform"
    loggamma: str = "loggamma"


class SelectDistOption(StrEnum):
    popular: str = "popular"
    full: str = "full"


class SelectDistStats(StrEnum):
    RSS: str = 'RSS'
    wasserstein: str = 'wasserstein'
    ks: str = 'ks'
    energy: str = 'energy'
    goodness_of_fit: str = 'goodness_of_fit'


class SelectDistMethod(StrEnum):
    parametric: str = 'parametric'
    quantile: str = 'quantile'
    percentile: str = 'percentile'
    discrete: str = 'discrete'


class SelectDistRequest(BaseModel):
    timeseries: Timeseries
    method: SelectDistMethod = Field(description="Метод нахождения распределения")
    distribution: Union[SelectDistOption, Distribution] = Field(description="Выбор распределения для перебора")
    statistics: SelectDistStats = Field(description="Выбор статистики для расчета")
    bins: int = Field(gt=0, default=20, description="Количество бинов распределений")


class SelectDistResult(BaseModel):
    name: Distribution
    score: float
    loc: float
    scale: float


class SelectDistResponse(BaseModel):
    best_dists: list[SelectDistResult]
    qq_result: QQResult
