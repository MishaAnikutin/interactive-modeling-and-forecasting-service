from typing import Annotated, Union, Literal

from pydantic import BaseModel, Field, ConfigDict, model_validator
from src.core.domain import Timeseries
from enum import Enum


class Distribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Exp(Distribution):
    name: Literal["exp"] = "exp"
    lambda_: float = Field(
        gt=0, le=1000,
        title="Параметр экспоненциального распределения",
        default=1,
    )

class Normal(Distribution):
    name: Literal["normal"] = "normal"
    mu_: float = Field(
        ge=-1e6, le=1e6,
        title="Среднее нормального распределения",
        default=0,
    )
    sigma_: float = Field(
        gt=0, le=1000,
        title="Стандартное отклонение нормального распределения",
        default=1,
    )

class Uniform(Distribution):
    name: Literal["uniform"] = "uniform"
    left_: float = Field(
        ge=-1e6, le=1e6,
        title="Левая граница равномерного распределения",
        default=0,
    )
    right_: float = Field(
        ge=-1e6, le=1e6,
        title="Правая граница равномерного распределения",
        default=1,
    )
    @model_validator(mode="after")
    def validate_boarders(self):
        if self.left_ >= self.right_:
            raise ValueError("Левая граница должна быть строго меньше правой")
        return self


DistributionUnion = Annotated[
    Union[
        Exp, Normal, Uniform
    ],
    Field(discriminator="name")
]

class DistributionEnum(str, Enum):
    normal = "normal"
    uniform = "uniform"
    exponential = "exponential"
    chi2 = "chi2"


class PPplotParams(BaseModel):
    timeseries: Timeseries
    distribution: DistributionEnum


class PPResult(BaseModel):
    theoretical_probs: list[float]
    empirical_probs: list[float]
    perfect_line_x: list[float]
    perfect_line_y: list[float]
