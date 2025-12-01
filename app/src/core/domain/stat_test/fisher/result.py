from typing import Optional

from pydantic import BaseModel, Field, model_validator
from src.core.domain.stat_test.conclusion import Conclusion
from src.shared.utils import validate_float_param


class FisherTestResult(BaseModel):
    p_value: Optional[float] = Field(..., title="P-значение")
    statistic: Optional[float] = Field(..., title="F-статистика")
    conclusion: Conclusion = Field(..., title="Заключение")

    @model_validator(mode="after")
    def validate_value(self):
        self.p_value = validate_float_param(self.p_value)
        self.statistic = validate_float_param(self.statistic)

        return self
