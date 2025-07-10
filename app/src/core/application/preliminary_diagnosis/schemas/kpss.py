from pydantic import BaseModel, Field
from enum import Enum

from src.core.application.preliminary_diagnosis.schemas.common import CriticalValues, StatTestParams


class RegressionEnum(str, Enum):
    ConstantOnly = 'c'
    ConstantAndTrend = 'ct'

class NlagsEnum(str, Enum):
    auto = 'auto'
    legacy = 'legacy'

class KpssParams(StatTestParams):
    regression: RegressionEnum = Field(default=RegressionEnum.ConstantOnly)
    nlags: NlagsEnum | int = Field(default=NlagsEnum.auto)

class KpssResult(BaseModel):
    p_value: float
    stat_value: float
    critical_values: CriticalValues
    lags: int