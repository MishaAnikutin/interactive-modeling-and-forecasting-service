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
    perfect_line_x: list[float] = Field(
        ...,
        title="Идеальная линия X",
        description="Координаты X идеальной линии y = x.",
    )
    perfect_line_y: list[float] = Field(
        ...,
        title="Идеальная линия Y",
        description="Координаты Y идеальной линии y = x.",
    )
