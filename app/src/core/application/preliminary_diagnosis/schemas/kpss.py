from pydantic import Field, model_validator
from enum import Enum

from src.core.application.preliminary_diagnosis.schemas.common import StatTestParams


class RegressionEnum(str, Enum):
    ConstantOnly = 'c'
    ConstantAndTrend = 'ct'

class NlagsEnum(str, Enum):
    auto = 'auto'
    legacy = 'legacy'

class KpssParams(StatTestParams):
    regression: RegressionEnum = Field(
        default=RegressionEnum.ConstantOnly,
        title="Компонента тренда, которую следует включить в тест"
    )
    nlags: NlagsEnum | int = Field(
        default=NlagsEnum.auto,
        title="Число лагов",
        description="Indicates the number of lags to be used. "
                    "If “auto” (default), lags is calculated using "
                    "the data-dependent method of Hobijn et al. (1998). "
                    "See also Andrews (1991), Newey & West (1994), "
                    "and Schwert (1989). "
                    "If set to “legacy”, uses int(12 * (n / 100)**(1 / 4)) , "
                    "as outlined in Schwert (1989)."
    )

    @model_validator(mode='after')
    def validate_nlags(self):
        if type(self.nlags) != NlagsEnum:
            if self.nlags < 0:
                raise ValueError("nlags must be non-negative")
        return self
