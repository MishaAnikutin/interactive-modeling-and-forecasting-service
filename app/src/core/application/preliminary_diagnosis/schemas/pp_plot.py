from pydantic import BaseModel, Field
from src.core.domain import Timeseries
from enum import Enum

from src.core.domain.distributions import Distribution


class PPplotParams(BaseModel):
    timeseries: Timeseries = Field(
        ...,
        title="Временной ряд",
        description="Временной ряд, для которого строится PP-plot.",
    )
    distribution: Distribution = Field(
        ...,
        title="Тип распределения",
        description="Теоретическое распределение, используемое для PP-plot.",
    )


class PPResult(BaseModel):
    theoretical_probs: list[float] = Field(
        ...,
        title="Теоретические вероятности",
        description="Вероятности, вычисленные из выбранного теоретического распределения.",
    )
    empirical_probs: list[float] = Field(
        ...,
        title="Эмпирические вероятности",
        description="Эмпирические вероятности, рассчитанные по данным временного ряда.",
    )
