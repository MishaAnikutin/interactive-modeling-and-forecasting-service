from datetime import datetime
from typing import Optional

from pydantic import BaseModel, model_validator, Field

from src.core.domain.stat_test.conclusion import Conclusion
from src.shared.utils import validate_float_param


class StudentTestResult(BaseModel):
    datetime: Optional[datetime]
    p_value: Optional[float] = Field(..., title="P-значение")
    statistic: Optional[float] = Field(..., title="T-статистика")
    conclusion: Conclusion = Field(..., title="Заключение")

    @model_validator(mode="after")
    def validate_value(self):
        self.p_value = validate_float_param(self.p_value)
        self.statistic = validate_float_param(self.statistic)

        return self
