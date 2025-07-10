from pydantic import BaseModel, Field, model_validator
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

    @model_validator(mode='after')
    def validate_nlags(self):
        if type(self.nlags) != NlagsEnum:
            if self.nlags < 0:
                raise ValueError("nlags must be non-negative")
        return self

class KpssResult(BaseModel):
    p_value: float
    stat_value: float
    critical_values: CriticalValues
    lags: int