from typing import Optional

from pydantic import Field
from enum import Enum

from src.core.application.preliminary_diagnosis.schemas.common import StatTestParams


class RegressionEnum(str, Enum):
    ConstantOnly = "c"
    TrendOnly = "t"
    ConstantAndTrend = "ct"


class AutoLagEnum(str, Enum):
    AIC = "aic"
    BIC = "bic"
    t_stat = "t-stat"


class ZivotAndrewsParams(StatTestParams):
    lags: Optional[int] = Field(default=None, ge=0)
    max_lags: Optional[int] = Field(default=None, ge=0)
    trim: float = Field(default=0.15, ge=0.0, le=0.33)
    regression: RegressionEnum = Field(default=RegressionEnum.ConstantOnly)
    autolag: AutoLagEnum = Field(default=AutoLagEnum.AIC)
