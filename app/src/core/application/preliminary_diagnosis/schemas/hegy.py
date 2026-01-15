from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from src.core.domain import Timeseries
from src.core.domain.stat_test.hegy import TrendType, CriteriaType


class HegyRequest(BaseModel):
    y: Timeseries
    max_lag: Optional[int] = Field(default=None, ge=0,
                                   description="максимальное число лагов в регрессии теста. Если `None`, то используется "
                                               "$12 {\left\lfloor\frac{T}{100}\right\rfloor}^{1/4}$")
    trend: TrendType = TrendType.NONE
    criteria: CriteriaType = CriteriaType.AIC
    S: int = Field(default=4, gt=1, description="Длина сезонного цикла")
    stats_only: bool = Field(default=False, description="Считать p-значения или нет")


class HegyStatisticType(str, Enum):
    zero_freq: str = 'Zero freq'
    nyquist: str = 'Nyquist'
    seas: str = 'Seas.'
    all: str = 'ALL'


class HegyStatistic(BaseModel):
    type: HegyStatisticType | str
    test_statistic: float
    p_value: float


class HegyResponse(BaseModel):
    statistics: List[HegyStatistic]
