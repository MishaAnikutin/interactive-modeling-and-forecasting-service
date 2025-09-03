from enum import Enum

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal

from src.core.application.preliminary_diagnosis.errors.acf_and_pacf import (
    InvalidMaxLagsError,
    LowCountObservationsError,
    InvalidAlphaError,
)
from src.core.domain import Timeseries


class AcfPacfMethod(str, Enum):
    ywunbiased: str = 'ywunbiased'
    ywadjusted: str = 'ywadjusted'
    ywmle: str = 'ywmle'
    ols: str = 'ols'
    ols_inefficient = 'ols-inefficient'
    ols_adjusted = 'ols-adjusted'
    ldadjusted = 'ldadjusted'
    ldbiased = 'ldbiased'
    burg = 'burg'



class AcfAndPacfRequest(BaseModel):
    ts: Timeseries = Field(..., description="Временной ряд")
    nlags: int = Field(
        default=12,
        ge=0,
        title="Количество лагов для расчета",
        description="Количество лагов для расчета ACF и PACF"
    )
    alpha: Optional[float] = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        title="Уровень значимости",
        description="Уровень значимости для доверительных интервалов"
    )
    pacf_method: AcfPacfMethod = Field(
        default=AcfPacfMethod.ywadjusted,
        description='Указывает, какой метод вычислений следует использовать'
    )

    @model_validator(mode='after')
    def validate_nlags(self):
        nobs = len(self.ts.values)
        if self.nlags >= nobs:
            raise ValueError(
                InvalidMaxLagsError(
                    detail=f"Количество лагов ({self.nlags}) не может превышать длину временного ряда ({nobs})"
                ).detail
            )

        if nobs <= self.nlags:
            raise ValueError(
                LowCountObservationsError(
                    detail=f"Недостаточно данных: требуется минимум {self.nlags + 1} наблюдений, получено {nobs}"
                ).detail
            )
        return self

    @model_validator(mode='after')
    def validate_alpha(self):
        if self.alpha is not None and (self.alpha <= 0 or self.alpha >= 1):
            raise ValueError(InvalidAlphaError().detail)
        return self


class AcfPacfResult(BaseModel):
    acf_values: List[float] = Field(..., description="Значения автокорреляционной функции")
    pacf_values: List[float] = Field(..., description="Значения частичной автокорреляционной функции")
    acf_confint: Optional[List[List[float]]] = Field(None, description="Доверительные интервалы ACF")
    pacf_confint: Optional[List[List[float]]] = Field(None, description="Доверительные интервалы PACF")

