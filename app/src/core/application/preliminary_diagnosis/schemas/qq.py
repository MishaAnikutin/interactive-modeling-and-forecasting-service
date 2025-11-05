from pydantic import BaseModel, Field
from src.core.domain import Timeseries
from src.core.domain.distributions import Distribution


class QQParams(BaseModel):
    timeseries: Timeseries
    distribution: Distribution


class QQResult(BaseModel):
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
